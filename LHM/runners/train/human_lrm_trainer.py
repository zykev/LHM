# -*- coding: utf-8 -*-
"""
HumanLRMTrainer — LHM-mini 在 4DDress 数据集上的训练 Runner。

使用 Huggingface Accelerate 实现分布式训练 + 混合精度。

训练入口：
  python -m LHM.launch train.human_lrm --config configs/training/human-lrm-mini-4ddress.yaml

多卡训练：
  accelerate launch --config_file configs/accelerate.yaml \\
      -m LHM.launch train.human_lrm \\
      --config configs/training/human-lrm-mini-4ddress.yaml
"""

import glob
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict

import kornia.metrics as kornia_metrics
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from LHM.losses import LPIPSLoss, PixelLoss
from LHM.losses.ball_loss import Heuristic_ASAP_Loss
from LHM.losses.offset_loss import Heuristic_ACAP_Loss
from LHM.runners import REGISTRY_RUNNERS
from LHM.runners.abstract import Runner

logger = get_logger(__name__)

# 固定的 key 列表，用于多卡聚合时构造定长 tensor。不能直接用本地 dict 的 key 集合，
# 因为各 loss 是否出现取决于动态权重调度/try-except（如 face_id），在某些 rank 上
# 可能和别的 rank 不一致；验证集样本数很少时，甚至某个 rank 分到的 shard 可能为空，
# 导致它的 agg dict 完全没有 key——这些情况下用不定长 tensor 做 all-reduce 会在
# 多卡间形状不一致而卡死/报错。
_LOSS_KEYS = ('masked_pixel', 'perceptual', 'mask', 'face_id', 'asap', 'acap')
_METRIC_KEYS = ('psnr', 'ssim', 'lpips')


# ===========================================================================
# 工具函数
# ===========================================================================

def parse_dynamic_weight(spec, global_step: int) -> float:
    """
    解析动态权重调度规格。
    格式: "start_step:start_val:end_val:end_step"  或  直接浮点数。
    在 [start_step, end_step] 区间内线性插值。
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if spec is None:
        return 0.0
    parts = str(spec).split(':')
    if len(parts) == 1:
        return float(parts[0])
    start_step, start_val, end_val, end_step = (
        int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
    )
    if global_step <= start_step:
        return start_val
    if global_step >= end_step:
        return end_val
    t = (global_step - start_step) / max(1, end_step - start_step)
    return start_val + t * (end_val - start_val)


def collate_fn_skip_none(batch):
    """
    去掉 batch 中加载失败（返回 None）的样本，再执行默认 collate。
    """
    from torch.utils.data.dataloader import default_collate
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def build_smplx_body_region_mapping(device='cuda') -> dict:
    """
    返回 SMPL-X 各身体区域对应的 GS 点索引映射。
    用于 Heuristic_ASAP_Loss / Heuristic_ACAP_Loss。
    此处为简化版——实际项目可替换为来自 human_model_files 的精确映射。
    """
    # 占位：按等比划分 20000 个 GS 点（与 dense_sample_pts 一致）
    N = 20000
    q, r = divmod(N, 4)
    head_idx  = list(range(0, q))
    upper_idx = list(range(q, 2*q))
    lower_idx = list(range(2*q, 3*q))
    hands_idx = list(range(3*q, N))
    return {
        'head':       head_idx,
        'upper_body': upper_idx,
        'lower_body': lower_idx,
        'hands':      hands_idx,
    }


# ===========================================================================
# 主 Trainer 类
# ===========================================================================

@REGISTRY_RUNNERS.register('train.human_lrm')
class HumanLRMTrainer(Runner):
    """LHM-mini 训练 Runner，适配 4DDress 多视角数据集。"""

    EXP_TYPE = 'lrm'

    def __init__(self):
        super().__init__()
        self.cfg = self._load_config()
        self._setup_exp_dir()
        self._setup_compile()
        self._setup_accelerator()
        self._setup_logger()
        self._setup_model()
        self._setup_datasets()
        self._setup_optimizer()
        self._setup_losses()
        self.global_step = 0
        self._resume_if_needed()

    def _setup_exp_dir(self):
        """统一实验目录：exps_root/<wandb name>/{logs,trackers,checkpoints,wandb}/。
        用 wandb 的 name 字段作为文件夹名，没配置 name 时回退到 experiment.parent/child。
        """
        exp = self.cfg.experiment
        wandb_name = self.cfg.logger.get('wandb', {}).get('name', None) or f'{exp.parent}_{exp.child}'
        exps_root = self.cfg.logger.get('exps_root', './exps')
        self.exp_dir = os.path.join(exps_root, wandb_name)
        os.makedirs(self.exp_dir, exist_ok=True)

    def _setup_compile(self):
        """将 cfg.compile 中的设置应用到 torch._dynamo（模型里的 @torch.compile
        装饰器在类定义时即生效，必须在构建模型前设置这些全局开关才有效）。"""
        cc = self.cfg.get('compile', {})
        if cc.get('disable', False):
            torch._dynamo.config.disable = True
        if cc.get('suppress_errors', False):
            torch._dynamo.config.suppress_errors = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
        if getattr(self, 'use_wandb', False):
            wandb.finish()

    # ── 配置加载 ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_config() -> DictConfig:
        """Load config plus evaluation-only dataset/checkpoint overrides."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True)
        parser.add_argument('--eval-only', action='store_true')
        parser.add_argument('--checkpoint', default=None,
                            help='local .pth checkpoint, or "pretrained" (LHM-MINI)')
        parser.add_argument('--save-render', action='store_true')
        parser.add_argument('--output-dir', default=None)
        parser.add_argument('--dataset-root', default=None,
                            help='raw dataset root used by an evaluation adapter')
        parser.add_argument('--metadata-root', default=None,
                            help='LHM-Track prepare output root used by an evaluation adapter')
        parser.add_argument('--sample-id', action='append',
                            help='evaluate one prepared sample id (can be repeated)')
        parser.add_argument('--sample-list', default=None,
                            help='newline-delimited prepared sample IDs to evaluate')
        args, _ = parser.parse_known_args()
        if args.sample_id and args.sample_list:
            parser.error('use --sample-id or --sample-list, not both')
        selected_ids = args.sample_id
        if args.sample_list:
            with open(args.sample_list) as handle:
                selected_ids = [line.split('#', 1)[0].strip() for line in handle]
            selected_ids = [item for item in selected_ids if item]
            if not selected_ids:
                parser.error('--sample-list is empty')
        cfg = OmegaConf.load(args.config)
        if args.dataset_root:
            cfg.dataset.raw_data_dir = args.dataset_root
        if args.metadata_root:
            for subset in cfg.dataset.subsets:
                subset.root_dirs = [args.metadata_root]
                subset.meta_path.train = os.path.join(args.metadata_root, 'label', 'train_list.json')
                subset.meta_path.val = os.path.join(args.metadata_root, 'label', 'val_list.json')
        if selected_ids is not None:
            cfg.dataset.eval_sample_ids = selected_ids
        cfg.runtime = {
            'eval_only': args.eval_only,
            'checkpoint': args.checkpoint,
            'save_render': args.save_render,
            'output_dir': args.output_dir,
        }
        return cfg

    # ── Accelerator / Logger ────────────────────────────────────────────────

    def _setup_accelerator(self):
        tc = self.cfg.train
        # 多卡训练用 DDP 时，模型里大量条件分支（use_face_id、latent_query_points_type
        # 分支、id_face_net 只在 w_fid>0 时用到等）很容易导致某些参数在某些 step 不
        # 参与反向传播——DDP 默认 find_unused_parameters=False 时会直接崩。这里把
        # train.find_unused_parameters 接到 DistributedDataParallelKwargs 上（单卡
        # /未用 accelerate launch 时这个 kwargs 不生效，无副作用）。
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=tc.get('find_unused_parameters', False)
        )
        self.accelerator = Accelerator(
            mixed_precision=tc.get('mixed_precision', 'bf16'),
            gradient_accumulation_steps=tc.get('accum_steps', 1),
            log_with=None,
            kwargs_handlers=[ddp_kwargs],
        )
        set_seed(self.cfg.experiment.get('seed', 42))

    def _setup_logger(self):
        log_root = os.path.join(self.exp_dir, 'logs')
        os.makedirs(log_root, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(name)s %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_root, 'train.log')),
                logging.StreamHandler(sys.stdout),
            ],
        )
        if self.accelerator.is_main_process:
            tracker_root = os.path.join(self.exp_dir, 'trackers')
            os.makedirs(tracker_root, exist_ok=True)
            trackers = self.cfg.logger.get('trackers', ['tensorboard'])
            if 'tensorboard' in trackers:
                self.writer = SummaryWriter(log_dir=tracker_root)
            else:
                self.writer = None
        else:
            self.writer = None

        self._setup_wandb()

    def _setup_wandb(self):
        """Initialize wandb on main process if 'wandb' is in cfg.logger.trackers."""
        trackers = self.cfg.logger.get('trackers', [])
        if 'wandb' not in trackers or not self.accelerator.is_main_process:
            self.use_wandb = False
            return
        if not _WANDB_AVAILABLE:
            logger.warning('wandb 未安装，跳过 wandb 初始化。pip install wandb')
            self.use_wandb = False
            return

        wc = self.cfg.logger.get('wandb', {})
        entity  = wc.get('entity', None) or None
        project = wc.get('project', 'lhm-training')
        name    = wc.get('name', None) or None
        tags    = list(wc.get('tags', []) or []) or None
        notes   = wc.get('notes', None) or None
        run_id  = wc.get('run_id', None) or None

        exp = self.cfg.experiment
        exp_name = f'{exp.parent}/{exp.child}'

        wandb.init(
            entity=entity,
            project=project,
            name=name or exp_name,
            tags=tags,
            notes=notes,
            id=run_id,
            resume='allow' if run_id else None,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            dir=self.exp_dir,  # wandb 会在此目录下创建 wandb/run-.../，与 logs/trackers/checkpoints 同级
        )
        self.use_wandb = True
        logger.info(f'wandb 初始化: project={project}, name={name or exp_name}')

    # ── 模型构建 ─────────────────────────────────────────────────────────────

    def _setup_model(self):
        from LHM.models import model_dict

        mc = self.cfg.model
        model_name = mc.get('model_name', 'human_lrm_sapdino_bh_sd3_5').lower()

        # 根据 model_name 选择模型类
        name_map = {
            'sapdiolrmbhsd3_5':                 'human_lrm_sapdino_bh_sd3_5',
            'human_lrm_sapdino_bh_sd3_5':       'human_lrm_sapdino_bh_sd3_5',
            'humanlrm':                          'human_lrm',
        }
        key = name_map.get(model_name.replace('_', '').lower(), 'human_lrm_sapdino_bh_sd3_5')

        model_cls = model_dict[key]

        # 将 OmegaConf 转换为标准 dict，过滤非模型参数的 key
        mc_dict = OmegaConf.to_container(mc, resolve=True)
        mc_dict.pop('model_name', None)

        runtime_ckpt = self.cfg.runtime.get('checkpoint', None)
        ckpt_path = runtime_ckpt or self.cfg.saver.get('load_model', None)
        if ckpt_path == 'pretrained':
            from LHM.utils.hf_hub import wrap_model_hub
            hf_model_cls = wrap_model_hub(model_cls)
            pretrained = hf_model_cls.from_pretrained('3DAIGC/LHM-MINI')
            # Keep the local 4D-Dress SMPL-X/PCA renderer configuration while
            # reusing the learned network weights from the official model.
            self.model = model_cls(**mc_dict)
            missing, unexpected = self.model.load_state_dict(pretrained.state_dict(), strict=False)
            logger.info('加载预训练模型: 3DAIGC/LHM-MINI')
            logger.info(f'预训练权重 missing={len(missing)}, unexpected={len(unexpected)}')
        else:
            self.model = model_cls(**mc_dict)
            if ckpt_path:
                if not os.path.isfile(ckpt_path):
                    raise FileNotFoundError(f'checkpoint does not exist: {ckpt_path}')
                state = torch.load(ckpt_path, map_location='cpu')
                state = state.get('model', state)
                state = {k.removeprefix('module.'): v for k, v in state.items()}
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                logger.info(f'加载权重 {ckpt_path}，missing={len(missing)}, unexpected={len(unexpected)}')

        # 统计参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable    = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'模型参数: 总计 {total_params/1e6:.1f}M, 可训练 {trainable/1e6:.1f}M')

    # ── 数据集 / DataLoader ──────────────────────────────────────────────────

    def _setup_datasets(self):
        from LHM.datasets.mixer import MixerDataset

        dc = self.cfg.dataset

        # 提取 dataset_kwargs（不含 subsets 等 Mixer 专属字段）
        dataset_kwargs = OmegaConf.to_container(dc, resolve=True)
        dataset_kwargs.pop('subsets', None)
        dataset_kwargs.pop('num_train_workers', None)
        dataset_kwargs.pop('num_val_workers', None)
        dataset_kwargs.pop('pin_mem', None)
        dataset_kwargs.pop('repeat_num', None)
        # 4D-Dress 使用该值将 crop/resize 后尺寸与 ViT patch 对齐。

        subsets = OmegaConf.to_container(dc.subsets, resolve=True)

        self.train_dataset = MixerDataset(
            split='train', subsets=subsets, **dataset_kwargs
        )
        self.val_dataset = MixerDataset(
            split='val', subsets=subsets,
            eval_all_views=self.cfg.runtime.get('eval_only', False),
            **dataset_kwargs,
        )

        num_train_workers = dc.get('num_train_workers', 4)
        num_val_workers   = dc.get('num_val_workers', 2)
        pin_mem = dc.get('pin_mem', True)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=num_train_workers,
            pin_memory=pin_mem,
            collate_fn=collate_fn_skip_none,
            drop_last=True,
            persistent_workers=num_train_workers > 0,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val.get('batch_size', 2),
            shuffle=False,
            num_workers=num_val_workers,
            pin_memory=pin_mem,
            collate_fn=collate_fn_skip_none,
            drop_last=False,
        )
        logger.info(f'训练集: {len(self.train_dataset)} 样本; '
                    f'验证集: {len(self.val_dataset)} 样本')

    # ── Optimizer / Scheduler ───────────────────────────────────────────────

    def _setup_optimizer(self):
        oc = self.cfg.train.optim

        if hasattr(self.model, 'obtain_params'):
            opt_groups = self.model.obtain_params(self.cfg)
        else:
            opt_groups = [{'params': self.model.parameters(), 'lr': oc.lr}]

        self.optimizer = torch.optim.AdamW(
            opt_groups,
            lr=oc.lr,
            betas=(oc.get('beta1', 0.9), oc.get('beta2', 0.95)),
            weight_decay=oc.get('weight_decay', 0.05),
        )

        sc = self.cfg.train.scheduler
        warmup_iters  = sc.get('warmup_real_iters', 3000)
        total_iters   = len(self.train_loader) * self.cfg.train.epochs
        sched_type    = sc.get('type', 'cosine')

        def lr_lambda(step):
            if step < warmup_iters:
                return step / max(1, warmup_iters)
            if sched_type == 'cosine':
                progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
                return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Accelerate 准备
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.val_loader,
            self.scheduler,
        )

    # ── 日志辅助方法 ─────────────────────────────────────────────────────────

    def _log_scalars(self, metrics: dict, prefix: str, step: int):
        """Log scalar metrics to TensorBoard and/or wandb."""
        if not self.accelerator.is_main_process:
            return
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f'{prefix}/{k}', v, step)
        if getattr(self, 'use_wandb', False):
            wandb.log({f'{prefix}/{k}': v for k, v in metrics.items()}, step=step)

    def _log_images_wandb(
        self,
        render_out: dict,
        batch: dict,
        batch_dev: dict,
        prefix: str = 'train',
        n_log: int = 2,
    ):
        """记录采样图片：wandb（若启用）+ 本地保存一份到
        exp_dir/images/<prefix>/step_<global_step>/ 下。"""
        if not self.accelerator.is_main_process:
            return

        B = render_out['comp_rgb'].shape[0]
        n = min(n_log, B)
        N = render_out['comp_rgb'].shape[1]  # N_tgt views

        # comp_rgb/comp_mask 在 gs_renderer.py 的 forward_animate_gs 里已经从
        # [B, N, H, W, 3]（channels last）permute 成 [B, N, 3/1, H, W]（channel first）
        # 再返回，这里不需要再 permute 一次。
        pred_rgb  = render_out['comp_rgb'][:n].detach().float().clamp(0, 1)
        pred_mask = (
            render_out['comp_mask'][:n].detach().float().clamp(0, 1)
            .expand(-1, -1, 3, -1, -1)
        )
        gt_images = batch_dev['render_images'][:n].detach().float().clamp(0, 1)

        # Flatten B×N into single batch dim for make_grid; nrow=N keeps views in one row
        pred_rgb_flat  = pred_rgb.reshape(n * N, *pred_rgb.shape[2:])
        pred_mask_flat = pred_mask.reshape(n * N, *pred_mask.shape[2:])
        gt_flat        = gt_images.reshape(n * N, *gt_images.shape[2:])

        # name -> (grid_tensor, caption)；先把所有要记录的图拼成 grid，
        # 再统一分发给 wandb.Image 和本地 save_image，避免重复计算。
        grids = {
            'render_rgb':  (make_grid(pred_rgb_flat.cpu(), nrow=N, padding=2), 'rendered RGB'),
            'render_mask': (make_grid(pred_mask_flat.cpu(), nrow=N, padding=2), 'rendered opacity/mask'),
            'gt_images':   (make_grid(gt_flat.cpu(), nrow=N, padding=2), 'GT target views'),
        }

        # Source images
        if 'src_images' in batch:
            src = batch['src_images'][:n, 0].detach().float().clamp(0, 1).cpu()
            grids['src_images'] = (make_grid(src, nrow=n, padding=2), 'source images')

        # Head crops
        if 'source_head_rgbs' in batch:
            head = batch['source_head_rgbs'][:n, 0].detach().float().clamp(0, 1).cpu()
            grids['head_crops'] = (make_grid(head, nrow=n, padding=2), 'head crops')

        # Depth map (colormap via matplotlib if available)
        if 'comp_depth' in render_out:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import numpy as np
                # comp_depth 是 [B, N, 1, H, W]（channel first，同 comp_rgb/comp_mask）
                depth = render_out['comp_depth'][:n, :, 0, :, :].detach().float().cpu()  # [n, N, H, W]
                depth_flat = depth.reshape(n * N, *depth.shape[2:])  # [n*N, H, W]
                # Normalize per-sample
                d_min, d_max = depth_flat.amin(dim=(1,2), keepdim=True), depth_flat.amax(dim=(1,2), keepdim=True)
                depth_norm = (depth_flat - d_min) / (d_max - d_min + 1e-6)
                cmap = plt.get_cmap('plasma')
                colored = torch.from_numpy(
                    np.stack([cmap(d.numpy())[:, :, :3] for d in depth_norm], axis=0)
                ).permute(0, 3, 1, 2).float()
                grids['render_depth'] = (make_grid(colored, nrow=N, padding=2), 'rendered depth')
            except Exception:
                pass

        # 本地保存一份到 exp_dir/images/<prefix>/step_<step>/<name>.png
        save_dir = os.path.join(self.exp_dir, 'images', prefix, f'step_{self.global_step:08d}')
        os.makedirs(save_dir, exist_ok=True)
        for name, (grid, _caption) in grids.items():
            save_image(grid, os.path.join(save_dir, f'{name}.png'))

        if getattr(self, 'use_wandb', False):
            imgs_log = {
                f'{prefix}/{name}': wandb.Image(grid, caption=caption)
                for name, (grid, caption) in grids.items()
            }
            # GS scaling histogram（只有 wandb 支持直方图，不存本地）
            if 'scaling_output' in render_out:
                scaling = render_out['scaling_output'][:n].detach().float().cpu().reshape(-1)
                imgs_log[f'{prefix}/gs_scaling_hist'] = wandb.Histogram(scaling.numpy())
            wandb.log(imgs_log, step=self.global_step)

    # ── 损失函数 ────────────────────────────────────────────────────────────

    def _setup_losses(self):
        device = self.accelerator.device
        lfc    = self.cfg.train.loss_func

        self.pixel_loss     = PixelLoss(option=lfc.get('pixel_loss', 'l1'))
        self.perceptual_loss = LPIPSLoss(device=device, prefech=False)

        # ASAP（ball） loss：限制 GS 缩放各向同性
        ball_cfg = lfc.get('ball_loss', {})
        if ball_cfg.get('type') == 'heuristic':
            group_mapping = build_smplx_body_region_mapping()
            self.ball_loss = Heuristic_ASAP_Loss(
                group_dict=OmegaConf.to_container(ball_cfg['group'], resolve=True),
                group_body_mapping=group_mapping,
            )
        else:
            from LHM.losses.ball_loss import ASAP_Loss
            self.ball_loss = ASAP_Loss()

        # ACAP（offset）loss：限制 GS 偏移量
        off_cfg = lfc.get('offset_loss', {})
        if off_cfg.get('type') in ('classical', 'heuristic'):
            group_mapping = build_smplx_body_region_mapping()
            self.offset_loss = Heuristic_ACAP_Loss(
                group_dict=OmegaConf.to_container(off_cfg['group'], resolve=True),
                group_body_mapping=group_mapping,
            )
        else:
            from LHM.losses.offset_loss import ACAP_Loss
            self.offset_loss = ACAP_Loss()

    # ── Checkpoint ──────────────────────────────────────────────────────────

    def _ckpt_dir(self) -> str:
        d = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(d, exist_ok=True)
        return d

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            return
        ckpt_dir  = self._ckpt_dir()
        ckpt_path = os.path.join(ckpt_dir, f'step_{self.global_step:08d}.pth')
        model_state = self.accelerator.unwrap_model(self.model).state_dict()
        torch.save(
            {
                'global_step': self.global_step,
                'model':       model_state,
                'optimizer':   self.optimizer.state_dict(),
                'scheduler':   self.scheduler.state_dict(),
            },
            ckpt_path,
        )
        logger.info(f'保存 checkpoint: {ckpt_path}')

        # 清理旧 checkpoint（保留最近 K 个）
        keep = self.cfg.saver.get('checkpoint_keep_level', 60)
        all_ckpts = sorted(
            glob.glob(os.path.join(ckpt_dir, 'step_*.pth'))
        )
        for old in all_ckpts[:-keep]:
            os.remove(old)

    def _resume_if_needed(self):
        if not self.cfg.saver.get('auto_resume', True):
            return
        ckpt_dir = self._ckpt_dir()
        all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'step_*.pth')))
        if not all_ckpts:
            return
        latest = all_ckpts[-1]
        state = torch.load(latest, map_location='cpu')
        self.global_step = state['global_step']
        self.accelerator.unwrap_model(self.model).load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        logger.info(f'从 checkpoint 恢复训练: {latest}，step={self.global_step}')

    # ── 损失计算 ─────────────────────────────────────────────────────────────

    def _compute_losses(self, render_out: dict, batch: dict) -> dict:
        """
        计算所有训练损失，返回各项 loss 的 dict（已乘权重）。
        """
        lc = self.cfg.train.loss
        gs  = self.global_step

        # GS3DRenderer.forward_animate_gs（gs_renderer.py）已经把 comp_rgb/comp_mask
        # 从 [B, N_tgt, H, W, 3]（channels last）permute 成 [B, N_tgt, 3, H, W]（channel
        # first）再返回，这里不需要再 permute 一次——否则会把 H/W 维度和 channel 维度搞混。
        pred_rgb   = render_out['comp_rgb'].contiguous()
        pred_mask  = render_out['comp_mask'].contiguous()  # [B, N, 1, H, W]

        gt_images = batch['render_images']   # [B, N, 3, H, W]
        gt_masks  = batch['render_masks']    # [B, N, 1, H, W]

        losses = {}

        # Masked pixel loss
        w_mpix = parse_dynamic_weight(lc.get('masked_pixel_weight', 1.0), gs)
        if w_mpix > 0:
            pred_masked = pred_rgb * gt_masks
            gt_masked   = gt_images * gt_masks
            losses['masked_pixel'] = w_mpix * self.pixel_loss(pred_masked, gt_masked)

        # Perceptual (LPIPS) loss
        w_perc = parse_dynamic_weight(lc.get('perceptual_weight', 1.0), gs)
        if w_perc > 0:
            losses['perceptual'] = w_perc * self.perceptual_loss(
                pred_rgb * gt_masks, gt_images * gt_masks, is_training=True
            )

        # Mask loss
        w_mask = parse_dynamic_weight(lc.get('mask_weight', 0), gs)
        if w_mask > 0:
            losses['mask'] = w_mask * F.binary_cross_entropy(
                pred_mask.clamp(1e-6, 1 - 1e-6), gt_masks
            )

        # Face ID loss（仅当模型使用 face_id 时）
        if self.cfg.model.get('use_face_id', False):
            w_fid = parse_dynamic_weight(lc.get('face_id_weight', 0), gs)
            if w_fid > 0:
                # 用源图像 head crop 和渲染的对应区域计算 face_id loss
                src_head = batch['source_head_rgbs'][:, 0]  # [B, 3, H, W]
                # 取每个 target view 渲染结果的顶部区域（近似头部）
                H = pred_rgb.shape[-2]
                head_h = H // 4
                pred_head = pred_rgb[:, :, :, :head_h, :]  # [B, N, 3, h, W]
                pred_head_flat = pred_head.reshape(-1, *pred_head.shape[2:])
                src_head_expand = src_head.unsqueeze(1).expand(
                    -1, pred_rgb.shape[1], -1, -1, -1
                ).reshape(-1, *src_head.shape[1:])
                src_head_expand = F.interpolate(
                    src_head_expand, size=pred_head_flat.shape[-2:], mode='bilinear', align_corners=False
                )
                try:
                    model_unwrap = self.accelerator.unwrap_model(self.model)
                    if hasattr(model_unwrap, 'id_face_net'):
                        feat_pred = model_unwrap.id_face_net(pred_head_flat)
                        feat_src  = model_unwrap.id_face_net(src_head_expand)
                        losses['face_id'] = w_fid * (
                            1 - F.cosine_similarity(feat_pred, feat_src, dim=-1).mean()
                        )
                except Exception:
                    pass

        # ASAP（ball）loss：控制 GS 各向同性
        w_ball = parse_dynamic_weight(lc.get('asap_weight', 0), gs)
        if w_ball > 0 and 'scaling_output' in render_out:
            # scaling_output: [B, N_pts, 3]（已 stack 自 gs_attrs_list）
            # Heuristic_ASAP_Loss 内部按 group_body_mapping 在 N_pts 维度上索引，
            # 必须保留 [B, N_pts, 3] 形状，不能把 B 和 N_pts 压平到一起。
            scaling = render_out['scaling_output']  # [B, N_pts, 3]
            losses['asap'] = w_ball * self.ball_loss(scaling)

        # ACAP（offset）loss：控制 GS 偏移量
        w_off = parse_dynamic_weight(lc.get('acap_weight', 0), gs)
        if w_off > 0 and 'offset_output' in render_out:
            offset = render_out['offset_output']  # [B, N_pts, 3]
            losses['acap'] = w_off * self.offset_loss(offset)

        return losses

    @torch.no_grad()
    def _compute_eval_metrics(self, render_out: dict, batch: dict) -> dict:
        """
        计算 PSNR / SSIM / LPIPS 评估指标（不加权、不参与反向传播，只用于监控）。

        逐视角裁到 mask 的 bounding box 再调 kornia/lpips 的库函数算（不同视角
        bbox 大小不同，没法整 batch 向量化，只能逐个算）。框内不需要再额外乘
        mask——dress4d_lhm.py 加载 GT 时已经把背景按 mask 替换成纯白，跟渲染器
        合成背景（render_bg_colors，纯白）一致，裁剪后直接比较即可。
        """
        pred_rgb  = render_out['comp_rgb'].contiguous().float().clamp(0, 1)
        gt_images = batch['render_images'].float().clamp(0, 1)
        gt_masks  = batch['render_masks'].float()

        B, N, C, H, W = pred_rgb.shape
        pred_flat = pred_rgb.reshape(B * N, C, H, W)
        gt_flat   = gt_images.reshape(B * N, C, H, W)
        mask_flat = gt_masks.reshape(B * N, 1, H, W)

        psnr_vals, ssim_vals, lpips_vals = [], [], []
        for i in range(pred_flat.shape[0]):
            coords = torch.nonzero(mask_flat[i, 0] > 0, as_tuple=False)
            if coords.numel() == 0:
                # 没有有效前景区域时保留原图
                crop_pred = pred_flat[i].unsqueeze(0)
                crop_gt   = gt_flat[i].unsqueeze(0)
            else:
                y0, x0 = coords.min(dim=0)[0]
                y1, x1 = coords.max(dim=0)[0]
                crop_pred = pred_flat[i, :, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)
                crop_gt   = gt_flat[i, :, y0:y1 + 1, x0:x1 + 1].unsqueeze(0)

            psnr_vals.append(kornia_metrics.psnr(crop_pred, crop_gt, max_val=1.0))
            ssim_vals.append(
                kornia_metrics.ssim(crop_pred, crop_gt, window_size=11, max_val=1.0).mean()
            )
            lpips_vals.append(
                self.perceptual_loss(
                    crop_pred.unsqueeze(1), crop_gt.unsqueeze(1), is_training=False
                )
            )

        return {
            'psnr':  torch.stack(psnr_vals).mean().item(),
            'ssim':  torch.stack(ssim_vals).mean().item(),
            'lpips': torch.stack(lpips_vals).mean().item(),
        }

    # ── 单步训练 ─────────────────────────────────────────────────────────────

    def _train_step(self, batch: dict, return_render_out: bool = False):
        """执行单个训练步。return_render_out=True 时额外返回 (render_out, batch_dev)。"""
        if batch is None:
            return ({}, None, None) if return_render_out else {}

        device = self.accelerator.device

        # 将 smplx_params 内的 tensor 移至设备
        smplx_params = {
            k: v.to(device) for k, v in batch['smplx_params'].items()
            if isinstance(v, torch.Tensor)
        }

        model_input = dict(
            image           = batch['src_images'].to(device),
            source_c2ws     = batch['source_c2ws'].to(device),
            source_intrs    = batch['source_intrs'].to(device),
            render_c2ws     = batch['render_c2ws'].to(device),
            render_intrs    = batch['render_intrs'].to(device),
            render_bg_colors= batch['render_bg_colors'].to(device),
            smplx_params    = smplx_params,
            # kwargs
            source_head_rgbs= batch['source_head_rgbs'].to(device),
            render_height   = batch['render_images'].shape[-2],
            render_width    = batch['render_images'].shape[-1],
            df_data         = None,
        )

        with self.accelerator.autocast():
            render_out = self.model(**model_input)

        # 移动 batch 图像到设备（计算 loss 时需要）
        batch_dev = {
            'render_images':     batch['render_images'].to(device),
            'render_masks':      batch['render_masks'].to(device),
            'source_head_rgbs':  batch['source_head_rgbs'].to(device),
        }

        losses = self._compute_losses(render_out, batch_dev)
        total_loss = sum(losses.values())

        with self.accelerator.accumulate(self.model):
            self.accelerator.backward(total_loss)

            if self.cfg.train.get('clip_grad_norm'):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.cfg.train.optim.clip_grad_norm
                )

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        loss_items = {k: v.item() for k, v in losses.items()}
        if return_render_out:
            return loss_items, render_out, batch_dev
        return loss_items

    # ── 验证 ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        device  = self.accelerator.device
        val_cfg = self.cfg.val
        n_debug = 0 if self.cfg.runtime.get('eval_only', False) else val_cfg.get('debug_batches', 10)
        img_mon = self.cfg.logger.get('image_monitor', {})
        n_log   = img_mon.get('samples_per_log', 2)

        agg = defaultdict(float)
        agg_metrics = defaultdict(float)
        count = 0
        first_render_out = None
        first_batch      = None
        first_batch_dev  = None

        for i, batch in enumerate(self.val_loader):
            if batch is None:
                continue
            if n_debug and i >= n_debug:
                break

            smplx_params = {
                k: v.to(device) for k, v in batch['smplx_params'].items()
                if isinstance(v, torch.Tensor)
            }
            with self.accelerator.autocast():
                render_out = self.model(
                    image            = batch['src_images'].to(device),
                    source_c2ws      = batch['source_c2ws'].to(device),
                    source_intrs     = batch['source_intrs'].to(device),
                    render_c2ws      = batch['render_c2ws'].to(device),
                    render_intrs     = batch['render_intrs'].to(device),
                    render_bg_colors = batch['render_bg_colors'].to(device),
                    smplx_params     = smplx_params,
                    source_head_rgbs = batch['source_head_rgbs'].to(device),
                    df_data          = None,
                )

            batch_dev = {
                'render_images':    batch['render_images'].to(device),
                'render_masks':     batch['render_masks'].to(device),
                'source_head_rgbs': batch['source_head_rgbs'].to(device),
            }
            losses = self._compute_losses(render_out, batch_dev)
            for k, v in losses.items():
                agg[k] += v.item() if isinstance(v, torch.Tensor) else v
            for k, v in self._compute_eval_metrics(render_out, batch_dev).items():
                agg_metrics[k] += v
            count += 1

            if self.cfg.runtime.get('save_render', False) and self.accelerator.is_main_process:
                self._save_eval_renders(render_out, batch)

            # Keep the first batch for image logging
            if first_render_out is None:
                first_render_out = {k: v.detach() for k, v in render_out.items()
                                    if isinstance(v, torch.Tensor)}
                first_batch     = batch
                first_batch_dev = batch_dev

        self.model.train()

        # accelerator.prepare() 之后 val_loader 会把验证集切分到各 rank，每个 rank
        # 上面这段循环只跑到了自己那一份 shard。这里用固定 key 的 tensor 做一次
        # all-reduce sum，把所有 rank 的 loss/metric 总和、样本数加起来，再统一在
        # 下面除以全局 count，否则多卡训练时打印/记录的验证结果只反映 rank0 那一份
        # 数据，而不是整个验证集。reduce 是 collective op，必须每个 rank 都无条件
        # 调用，不能放在 count==0 的 early return 之后。
        stats = torch.tensor(
            [agg.get(k, 0.0) for k in _LOSS_KEYS]
            + [agg_metrics.get(k, 0.0) for k in _METRIC_KEYS]
            + [float(count)],
            device=device,
        )
        stats = self.accelerator.reduce(stats, reduction='sum')
        n_loss = len(_LOSS_KEYS)
        global_count = stats[-1].item()

        if global_count == 0:
            return {}

        avg = {k: stats[i].item() / global_count for i, k in enumerate(_LOSS_KEYS)}
        avg['total'] = sum(avg.values())
        avg.update({
            k: stats[n_loss + i].item() / global_count for i, k in enumerate(_METRIC_KEYS)
        })

        self._log_scalars(avg, 'val', self.global_step)

        if first_render_out is not None:
            self._log_images_wandb(
                first_render_out, first_batch, first_batch_dev,
                prefix='val', n_log=n_log,
            )

        logger.info(f'[val step={self.global_step}] ' +
                    ', '.join(f'{k}={v:.4f}' for k, v in avg.items()))
        return avg

    @torch.no_grad()
    def _save_eval_renders(self, render_out: dict, batch: dict):
        """Save prediction-only renders; GT/source/masks are intentionally omitted."""
        root = self.cfg.runtime.get('output_dir') or os.path.join(self.exp_dir, 'eval')
        renders = render_out['comp_rgb'].detach().float().cpu()
        uids = batch.get('uid', [f'sample_{i:05d}' for i in range(renders.shape[0])])
        if isinstance(uids, str):
            uids = [uids]
        for batch_idx, uid in enumerate(uids):
            sample_dir = os.path.join(root, 'renders', str(uid))
            os.makedirs(sample_dir, exist_ok=True)
            view_ids = batch.get('render_view_ids', [f'{index:02d}' for index in range(renders.shape[1])])
            if isinstance(view_ids, list):
                view_ids = [value[batch_idx] if isinstance(value, (list, tuple)) else value for value in view_ids]
            for view_idx, image in enumerate(renders[batch_idx]):
                save_image(image.clamp(0, 1), os.path.join(sample_dir, f'{view_ids[view_idx]}.png'))

    # ── 主训练循环 ────────────────────────────────────────────────────────────

    def run(self):
        if self.cfg.runtime.get('eval_only', False):
            metrics = self._validate()
            if self.accelerator.is_main_process:
                root = self.cfg.runtime.get('output_dir') or os.path.join(self.exp_dir, 'eval')
                os.makedirs(root, exist_ok=True)
                with open(os.path.join(root, 'metrics.json'), 'w', encoding='utf-8') as handle:
                    json.dump(metrics, handle, indent=2)
                logger.info(f'evaluation metrics: {os.path.join(root, "metrics.json")}')
            return

        tc        = self.cfg.train
        val_cfg   = self.cfg.val
        saver_cfg = self.cfg.saver

        self.model.train()
        log_interval  = self.cfg.logger.get('log_global_steps', 50)
        ckpt_period   = saver_cfg.get('checkpoint_global_steps', 1000)
        val_period    = val_cfg.get('global_step_period', 1000)
        total_steps   = len(self.train_loader) * tc.epochs

        logger.info(f'开始训练，总步数约 {total_steps}，当前 step={self.global_step}')
        t0 = time.time()

        for epoch in range(tc.epochs):
            for batch in self.train_loader:
                # hyper_step 是模型自定义方法，accelerator.prepare() 后 self.model
                # 可能被 DistributedDataParallel 包装，需要 unwrap 才能调用自定义方法。
                self.accelerator.unwrap_model(self.model).hyper_step(self.global_step)

                # 训练阶段只记录 loss/metrics，不记录图片（图片只在 _validate() 里记录）
                should_log_scalars = (
                    (self.global_step + 1) % log_interval == 0
                    and self.accelerator.is_main_process
                )
                step_result = self._train_step(batch, return_render_out=should_log_scalars)

                if should_log_scalars:
                    losses, render_out, batch_dev = step_result
                else:
                    losses = step_result
                    render_out = batch_dev = None

                if not losses:
                    continue
                self.global_step += 1

                # 标量日志
                if self.global_step % log_interval == 0 and self.accelerator.is_main_process:
                    total = sum(losses.values())
                    elapsed = time.time() - t0
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(
                        f'[step {self.global_step}/{total_steps}] '
                        f'loss={total:.4f} lr={lr:.2e} elapsed={elapsed:.0f}s'
                    )
                    scalar_metrics = dict(losses)
                    scalar_metrics['loss_total'] = total
                    scalar_metrics['lr'] = lr
                    if render_out is not None:
                        scalar_metrics.update(self._compute_eval_metrics(render_out, batch_dev))
                    self._log_scalars(scalar_metrics, 'train', self.global_step)

                # 验证（loss/metrics + 采样图片）
                if self.global_step % val_period == 0:
                    self._validate()

                # Checkpoint（存档前后加 barrier，避免其他 rank 在主进程写盘时
                # 跑到下一轮反向传播，导致各 rank step 不同步）
                if self.global_step % ckpt_period == 0:
                    self.accelerator.wait_for_everyone()
                    self._save_checkpoint()
                    self.accelerator.wait_for_everyone()

        # 训练结束保存最终 checkpoint
        self.accelerator.wait_for_everyone()
        self._save_checkpoint()
        logger.info('训练完成。')
