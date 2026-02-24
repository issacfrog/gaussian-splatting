#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

# 渲染函数
# viewpoint_camera 当前渲染视角 
# GaussianModel 当前高斯模型
# pipe 用来控制一些参数是否进行学习等
# bg_color 背景颜色
# scaling_modifier 缩放因子
# separate_sh 是否分离SH
# override_color 是否覆盖颜色
# use_trained_exp 是否使用训练好的曝光参数
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # step1
    # 用来接收3D高斯投影到2D坐标的可微通道
    # 3DGS本身是使用系数的高斯球来表征稠密的空间
    # 而后将高斯球投影到2D坐标系中，并将其存储在screenspace_points中
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad() # 强制保存中间梯度，用来后续访问
    except:
        pass

    # step2 构造Rasterization Settings 栅格化
    # 从相机 FoV 计算 tan(fov/2)，并构造光栅化配置。
    # 这些参数共同决定了高斯如何从世界坐标投影到图像平面。
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # 
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # 实例化光栅
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 3) 准备几何相关输入。 输入的是高斯模型
    # means3D 是可学习的高斯中心；means2D 是屏幕空间梯度通道；opacity 为可学习透明度。
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # 4) 协方差的两种路径：
    # - Python 预计算协方差（cov3D_precomp）
    # - 传入 scaling/rotation 让 CUDA rasterizer 内部计算
    scales = None
    rotations = None
    cov3D_precomp = None

    # 协方差的计算方式
    # 使用python预计算或cuda内部计算
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # 5) 颜色的三种来源：
    # - override_color：外部直接覆盖颜色 调试用
    # - convert_SHs_python=True：在 Python 中把 SH 系数转换成 RGB
    # - 默认：把 SH 系数传给 rasterizer，在 CUDA 内完成 SH->RGB 在cuda中处理，更快
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # 6) 调用 CUDA rasterizer 前向：得到渲染图像和每个高斯在屏幕上的半径。
    # 当前代码适配的是旧版 rasterizer，返回 (rendered_image, radii)，不含 depth。
    # 1.执行3D到2D的投影
    # 2.协方差投影 
    # 返回渲染后的图片和每个高斯在2D平面上的半径
    # 注rasterizer的GaussianRasterizer类继承了nn.module，默认的其会调用类中的forward函数
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
        
    # 7) 可选曝光仿射变换（仅在训练曝光参数时启用）。
    # 对每张图应用其对应 3x4 exposure 矩阵，提升跨视角亮度一致性。
    # 用来解决跨视角亮度不一致的问题，防止模型将曝光差异学到球谐函数中
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # 8) 后处理与返回：
    # - clamp 到 [0,1]
    # - visibility_filter 表示当前视角可见的高斯（radii > 0）
    #   训练时用它更新 max_radii2D 与 densification 统计
    rendered_image = rendered_image.clamp(0, 1) # 将渲染结果限制在0~1之间，因为真实物理是这样的，但是训练出来的不一定满足
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points, # 主要是为了将梯度传递出去给梯度统计等处理用
        "visibility_filter" : radii > 0,
        "radii": radii,
        }
    
    return out
