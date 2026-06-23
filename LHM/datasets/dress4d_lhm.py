# -*- coding: utf-8 -*-
"""
Dress4DLHMDataset — 4DDress 多视角数据集适配 LHM 训练格式。

依赖 prepare_4ddress.py 的输出：
  {root}/{uid}/smplx_params/{frame:05d}.json   — 前视角相机坐标系 SMPL-X
  {root}/{uid}/cameras.json                    — 所有相机内外参
  {root}/{uid}/flame_params/{frame:05d}.json   — FLAME bbox（可选，用于 head crop）

原始图像从 raw_data_dir 读取：
  {raw_data_dir}/{subject}/{outfit}/{take}/Capture/{cam_id}/images/*.png
  {raw_data_dir}/{subject}/{outfit}/{take}/Capture/{cam_id}/masks/*.png
"""

import glob
import json
import os
import random

import cv2
import numpy as np
import torch

from .base import BaseDataset

CAMERA_VIEWS = ['0004', '0028', '0052', '0076']
ASPECT_HW = 5.0 / 3.0  # LHM 期望的高:宽


# ---------------------------------------------------------------------------
# 图像/内参工具
# ---------------------------------------------------------------------------

def _find_capture_file(cam_raw_dir: str, frame_idx: int, subdir: str) -> str:
    """构造 4DDress Capture/{cam}/{subdir} 下的文件路径。
    命名规则：images/capture-f{frame:05d}.png，masks/mask-f{frame:05d}.png。
    """
    prefix = 'capture' if subdir == 'images' else 'mask'
    path = os.path.join(cam_raw_dir, subdir, f'{prefix}-f{frame_idx:05d}.png')
    return path if os.path.exists(path) else ''


# ---------------------------------------------------------------------------
# 跟 LHM 官方 runners/infer/utils.py:preprocess_image 对齐的预处理流程
# （resize_image_keepaspect_np / center_crop_according_to_mask /
#  calc_new_tgt_size_by_aspect 的训练数据版本，逐帧用真实 mask 动态裁剪，
#  而不是像旧版 _crop_params 那样只按相机的静态分辨率做几何中心裁剪）
# ---------------------------------------------------------------------------

MAX_TGT_SIZE = 896  # 与官方 human_lrm.py 里硬编码的 max_tgt_size 保持一致


def _resize_keepaspect(img: np.ndarray, max_tgt_size: int):
    """按最长边等比缩放到 max_tgt_size，返回 (resized_img, ratio)。"""
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), ratio


def _center_crop_by_mask(img: np.ndarray, mask: np.ndarray, cx: float, cy: float,
                          aspect_standard: float, enlarge_ratio=(1.0, 1.0)):
    """以相机真实光轴位置 (cx, cy) 为基准（不是图像几何中心！），裁出"刚好包住
    整个 mask、且满足目标纵横比"的区域，裁剪框只会被撑大以满足纵横比，不会缩小
    到比 mask 包围盒还小——保证不裁到人。

    GS3DRenderer 的投影矩阵（getProjectionMatrix/intrinsic_to_fov）是完全对称的
    视锥，只用 fx,fy 算 FoV，根本不读 cx,cy，也就是说渲染器只能正确渲染"光轴正
    好在画面中心"的相机。如果围绕图像几何中心裁剪，再在 _load_view 里把 cx,cy
    强制设成新图像中心，等于无视了真实标定的 cx,cy 跟几何中心之间的差异，凭空
    引入一个恒定的像素偏移（4DDress 真实相机的 cx,cy 并不在图像几何中心）。围绕
    真实 cx,cy 裁剪，才能保证裁完之后真实光轴恰好落在新图像中心，让后面"强制
    居中"是一个几乎无误差的精确陈述，而不是引入偏移。

    返回 (cropped_img, cropped_mask, offset_x, offset_y)。
    """
    ys, xs = np.where(mask > 0)
    H, W = img.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        # 没有有效前景区域，没法用 mask 定位，退化成保留原图（不裁剪）
        return img, mask, 0, 0

    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    cx_i, cy_i = int(round(cx)), int(round(cy))

    half_w = max(abs(cx_i - x_min), abs(cx_i - x_max))
    half_h = max(abs(cy_i - y_min), abs(cy_i - y_max))
    half_w_raw, half_h_raw = half_w, half_h
    aspect = half_h / max(half_w, 1)

    if aspect >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    # 不超出原图边界：注意 cx,cy 不在几何中心，左右/上下可用空间不对称
    max_half_w = min(cx_i, W - cx_i)
    max_half_h = min(cy_i, H - cy_i)
    if half_h > max_half_h:
        half_w = round(half_h_raw / aspect_standard)
        half_h = half_h_raw
    if half_w > max_half_w:
        half_h = round(half_w_raw * aspect_standard)
        half_w = half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_min, enlarge_max = enlarge_ratio
        enlarge_max_real = min(max_half_h / max(half_h, 1), max_half_w / max(half_w, 1))
        enlarge_max = min(enlarge_max_real, enlarge_max)
        enlarge_min = min(enlarge_max_real, enlarge_min)
        cur = np.random.rand() * (enlarge_max - enlarge_min) + enlarge_min
        half_h, half_w = round(cur * half_h), round(cur * half_w)

    half_h = min(half_h, max_half_h)
    half_w = min(half_w, max_half_w)

    offset_x = cx_i - half_w
    offset_y = cy_i - half_h
    cropped_img  = img[offset_y:offset_y + 2 * half_h, offset_x:offset_x + 2 * half_w]
    cropped_mask = mask[offset_y:offset_y + 2 * half_h, offset_x:offset_x + 2 * half_w]
    return cropped_img, cropped_mask, offset_x, offset_y


def _calc_tgt_hw(aspect_standard: float, tgt_w: int, multiply: int):
    """目标 (H, W)：H = tgt_w * aspect_standard，两边都向下取整到 multiply 的
    整数倍（跟 ViT patch size 对齐，避免 patch embedding 卷积在边缘截断）。"""
    tgt_h = tgt_w * aspect_standard
    tgt_h = max(int(tgt_h / multiply) * multiply, multiply)
    tgt_w = max(int(tgt_w / multiply) * multiply, multiply)
    return tgt_h, tgt_w


def _load_view(img_path: str, mask_path: str, K_raw: np.ndarray, target_w: int,
               multiply: int, aspect_standard: float = ASPECT_HW,
               max_tgt_size: int = MAX_TGT_SIZE, enlarge_ratio=(1.0, 1.0)):
    """完整复刻 LHM 官方 preprocess_image 的预处理流程：
    1. 读图 + mask，背景合成纯白（跟渲染器 render_bg_colors 一致）
    2. 按最长边等比缩放到 max_tgt_size，统一不同原始分辨率下人物的像素尺度
    3. 用 mask 包围盒、围绕图像几何中心裁出恰好包住人、满足目标纵横比的区域
       （旧版 _crop_params 只按相机的静态分辨率裁固定窗口，不看人在画面里的
       实际位置/姿态，可能裁到人；这里跟官方一样动态保证不裁到人）
    4. resize 到最终输出尺寸（取整到 multiply 的整数倍）
    5. 相机内参跟每一步同步变换，最终强制主点居中（此时已非常接近图像中心，
       跟官方在 utils.py 里的断言一致，强制居中不会引入明显误差）
    返回 (img_tensor[3,H,W], mask_tensor[1,H,W], intr_4x4)。
    """
    tgt_h, tgt_w = _calc_tgt_hw(aspect_standard, target_w, multiply)

    img_bgr = cv2.imread(img_path) if img_path else None
    if img_bgr is None:
        img_t  = torch.ones(3, tgt_h, tgt_w, dtype=torch.float32)
        mask_t = torch.zeros(1, tgt_h, tgt_w, dtype=torch.float32)
        intr = torch.eye(4, dtype=torch.float32)
        intr[0, 2], intr[1, 2] = tgt_w / 2, tgt_h / 2
        return img_t, mask_t, intr

    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = (mask_gray > 127).astype(np.float32) if mask_gray is not None \
        else np.ones(img.shape[:2], dtype=np.float32)

    # 背景合成纯白
    img = img * mask[:, :, None] + 1.0 * (1 - mask[:, :, None])

    K = K_raw.astype(np.float64).copy()

    # 1) 按最长边缩放到 max_tgt_size
    img, ratio0 = _resize_keepaspect(img, max_tgt_size)
    mask, _     = _resize_keepaspect(mask, max_tgt_size)
    K[0, 0] *= ratio0; K[0, 2] *= ratio0
    K[1, 1] *= ratio0; K[1, 2] *= ratio0

    # 2) 用 mask 包围盒裁剪，保证不裁到人；围绕相机真实光轴 (K[0,2],K[1,2])，
    # 不是图像几何中心（见 _center_crop_by_mask 注释）
    img, mask, off_x, off_y = _center_crop_by_mask(
        img, mask, K[0, 2], K[1, 2], aspect_standard, enlarge_ratio
    )
    K[0, 2] -= off_x
    K[1, 2] -= off_y

    # 3) resize 到最终目标尺寸
    cur_h, cur_w = img.shape[:2]
    ratio_y, ratio_x = tgt_h / cur_h, tgt_w / cur_w
    img  = cv2.resize(img,  (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
    K[0, 0] *= ratio_x; K[0, 2] *= ratio_x
    K[1, 1] *= ratio_y; K[1, 2] *= ratio_y

    # 4) 强制主点居中
    intr = torch.eye(4, dtype=torch.float32)
    intr[0, 0] = float(K[0, 0])
    intr[1, 1] = float(K[1, 1])
    intr[0, 2] = tgt_w / 2
    intr[1, 2] = tgt_h / 2

    img_t  = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    mask_t = torch.from_numpy((mask > 0.5).astype(np.float32)).unsqueeze(0)
    return img_t, mask_t, intr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dress4DLHMDataset(BaseDataset):
    """
    4DDress 数据集适配 LHM 训练。每个样本为一帧多视角数据：
      source: 前视角图像 + head crop
      target: 其余 N_tgt 视角图像 + mask（渲染监督）
      SMPL-X 在前视角相机坐标系下，target c2w 为相对 source 的变换。

    Args:
        root_dirs:         已准备数据的根目录列表（prepare_4ddress.py 输出）
        meta_path:         train_list.json / val_list.json 路径
        raw_data_dir:      4DDress 原始数据路径（读取多视角图像用）
        source_image_res:  source 图像宽度，默认 512
        render_image:      {"low": 384} 渲染目标宽度
        sample_side_views: 目标视角数，最多 3
        use_flame:         是否加载 FLAME 参数
        src_head_size:     head crop 边长，默认 112
        multiply:          最终图像尺寸向下取整到的倍数，跟 ViT patch size 对齐
        max_tgt_size:      预裁剪前的最长边缩放尺寸，对齐官方 preprocess_image
        enlarge_ratio:      mask 裁剪框的随机放大范围，[1.0, 1.0] 即不做增强
    """

    def __init__(
        self,
        root_dirs,
        meta_path: str,
        raw_data_dir: str = '',
        source_image_res: int = 512,
        render_image: dict = None,
        sample_side_views: int = 3,
        use_flame: bool = False,
        src_head_size: int = 112,
        multiply: int = 16,
        max_tgt_size: int = MAX_TGT_SIZE,
        enlarge_ratio=(1.0, 1.0),
        **kwargs,
    ):
        super().__init__(root_dirs, meta_path)
        self.root_dirs        = [root_dirs] if isinstance(root_dirs, str) else list(root_dirs)
        self.raw_data_dir     = raw_data_dir
        self.source_image_res = source_image_res
        self.render_image_res = (render_image or {}).get('high', (render_image or {}).get('low', 384))
        self.sample_side_views = sample_side_views
        self.multiply          = multiply
        self.max_tgt_size       = max_tgt_size
        self.enlarge_ratio      = tuple(enlarge_ratio)
        self.use_flame        = use_flame
        self.src_head_size    = src_head_size

    def _find_prepared_dir(self, uid: str) -> str:
        for rd in self.root_dirs:
            d = os.path.join(rd, uid)
            if os.path.isdir(d):
                return d
        raise FileNotFoundError(f'找不到 uid={uid} 的已处理目录')

    def _raw_item_dir(self, uid: str) -> str:
        """uid = 'subject_outfit_take' → {raw_data_dir}/{subject}/{outfit}/{take}"""
        parts = uid.rsplit('_', 2)
        if len(parts) != 3:
            raise ValueError(f'无法解析 uid: {uid}')
        subject, outfit, take = parts
        return os.path.join(self.raw_data_dir, subject, outfit, take)

    def _crop_head(self, img_path: str, prepared_dir: str, frame_idx: int) -> torch.Tensor:
        """从 FLAME bbox 裁剪头部，返回 [3, head_size, head_size]；无 bbox 时返回零张量。"""
        hs = self.src_head_size
        flame_path = os.path.join(prepared_dir, 'flame_params', f'{frame_idx:05d}.json')
        if not os.path.exists(flame_path):
            return torch.zeros(3, hs, hs, dtype=torch.float32)
        try:
            with open(flame_path) as f:
                x1, y1, x2, y2 = [int(v) for v in json.load(f)['bbox']]
            img = cv2.imread(img_path)
            if img is None:
                return torch.zeros(3, hs, hs, dtype=torch.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return torch.zeros(3, hs, hs, dtype=torch.float32)
            head = cv2.resize(img[y1:y2, x1:x2], (hs, hs), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(head.astype(np.float32) / 255.0).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, hs, hs, dtype=torch.float32)

    def inner_get_item(self, idx: int) -> dict:
        uid = self.uids[idx]
        prepared_dir = self._find_prepared_dir(uid)

        # --- cameras.json ---
        with open(os.path.join(prepared_dir, 'cameras.json')) as f:
            cameras = json.load(f)
        sorted_cams = sorted(cameras.items(), key=lambda x: x[1]['sorted_idx'])
        src_cam_id  = sorted_cams[0][0]
        # 所有 4 个相机均作为 target（含前视角自身，提供自重建监督）
        tgt_cam_ids = [cid for cid, _ in sorted_cams]
        src_cam     = cameras[src_cam_id]
        N_tgt       = len(tgt_cam_ids)

        # --- 随机帧 ---
        json_files = sorted(glob.glob(os.path.join(prepared_dir, 'smplx_params', '*.json')))
        if not json_files:
            raise FileNotFoundError(f'smplx_params 为空: {prepared_dir}')
        chosen = random.choice(json_files)
        frame_idx = int(os.path.splitext(os.path.basename(chosen))[0])

        # --- SMPL-X 参数 ---
        with open(chosen) as f:
            sp = json.load(f)

        def _t(key, shape):
            return torch.tensor(sp[key], dtype=torch.float32).reshape(shape)

        betas     = _t('betas',     (10,))
        root_pose = _t('root_pose', (3,))
        body_pose = _t('body_pose', (21, 3))
        jaw_pose  = _t('jaw_pose',  (3,))
        leye_pose = _t('leye_pose', (3,))
        reye_pose = _t('reye_pose', (3,))
        lhand     = _t('lhand_pose', (15, 3))
        rhand     = _t('rhand_pose', (15, 3))
        expr      = _t('expr',       (100,))
        trans     = _t('trans',      (3,))

        def _expand(t):
            return t.unsqueeze(0).expand(N_tgt, *t.shape).clone()

        smplx_params = {
            'root_pose':  _expand(root_pose),
            'body_pose':  _expand(body_pose),
            'jaw_pose':   _expand(jaw_pose),
            'leye_pose':  _expand(leye_pose),
            'reye_pose':  _expand(reye_pose),
            'lhand_pose': _expand(lhand),
            'rhand_pose': _expand(rhand),
            'expr':       _expand(expr),
            'trans':      _expand(trans),
            'betas':      betas,
        }

        # --- 相机外参（相对 c2w）---
        c2w_src = np.array(src_cam['c2w'], dtype=np.float64)
        w2c_src = np.linalg.inv(c2w_src)
        render_c2ws = torch.stack([
            torch.tensor(w2c_src @ np.array(cameras[cid]['c2w'], dtype=np.float64), dtype=torch.float32)
            for cid in tgt_cam_ids
        ])  # [N_tgt, 4, 4]
        source_c2ws = torch.eye(4).unsqueeze(0)  # [1, 4, 4]

        render_bg_colors = torch.ones(N_tgt, 3)

        # --- 图像 + mask + 内参：逐帧用 _load_view 动态裁剪（见函数注释），
        # 内参由裁剪/缩放过程同步算出，不能像旧版那样跟图像加载分开算 ---
        if self.raw_data_dir:
            raw_item = self._raw_item_dir(uid)
            src_cam_raw  = os.path.join(raw_item, 'Capture', src_cam_id)
            src_img_path  = _find_capture_file(src_cam_raw, frame_idx, 'images')
            src_mask_path = _find_capture_file(src_cam_raw, frame_idx, 'masks')
        else:
            src_img_path  = os.path.join(prepared_dir, 'imgs_png', f'{frame_idx:05d}.png')
            src_mask_path = ''

        src_image, _src_mask, source_intr = _load_view(
            src_img_path, src_mask_path, np.array(src_cam['intrinsics']),
            self.source_image_res, self.multiply,
            max_tgt_size=self.max_tgt_size, enlarge_ratio=self.enlarge_ratio,
        )
        src_image    = src_image.unsqueeze(0)     # [1, 3, H_src, W_src]
        source_intrs = source_intr.unsqueeze(0)   # [1, 4, 4]

        render_images, render_masks, render_intrs = [], [], []
        for cid in tgt_cam_ids:
            if self.raw_data_dir:
                tgt_cam_raw   = os.path.join(raw_item, 'Capture', cid)
                tgt_img_path  = _find_capture_file(tgt_cam_raw, frame_idx, 'images')
                tgt_mask_path = _find_capture_file(tgt_cam_raw, frame_idx, 'masks')
            else:
                tgt_img_path, tgt_mask_path = '', ''
            img, mask, intr = _load_view(
                tgt_img_path, tgt_mask_path, np.array(cameras[cid]['intrinsics']),
                self.render_image_res, self.multiply,
                max_tgt_size=self.max_tgt_size, enlarge_ratio=self.enlarge_ratio,
            )
            render_images.append(img)
            render_masks.append(mask)
            render_intrs.append(intr)

        render_images = torch.stack(render_images)  # [N_tgt, 3, H, W]
        render_masks  = torch.stack(render_masks)   # [N_tgt, 1, H, W]
        render_intrs  = torch.stack(render_intrs)   # [N_tgt, 4, 4]

        src_head_rgb = self._crop_head(src_img_path, prepared_dir, frame_idx).unsqueeze(0)

        flame_params = None
        if self.use_flame:
            fp = os.path.join(prepared_dir, 'flame_params', f'{frame_idx:05d}.json')
            if os.path.exists(fp):
                with open(fp) as f:
                    flame_params = json.load(f)

        sample = {
            'src_images':       src_image,        # [1, 3, H_src, W_src]
            'source_head_rgbs': src_head_rgb,     # [1, 3, hs, hs]
            'render_images':    render_images,    # [N_tgt, 3, H, W]
            'render_masks':     render_masks,     # [N_tgt, 1, H, W]
            'render_c2ws':      render_c2ws,      # [N_tgt, 4, 4]
            'render_intrs':     render_intrs,     # [N_tgt, 4, 4]
            'source_c2ws':      source_c2ws,      # [1, 4, 4]
            'source_intrs':     source_intrs,     # [1, 4, 4]
            'render_bg_colors': render_bg_colors, # [N_tgt, 3]
            'smplx_params':     smplx_params,
            'uid':              uid,
            'frame_idx':        frame_idx,
        }
        if flame_params is not None:
            sample['flame_params'] = flame_params
        return sample
