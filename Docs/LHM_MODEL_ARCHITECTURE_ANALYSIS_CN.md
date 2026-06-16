# LHM 模型结构、张量维度与规模分析

本文从 `LHM/models/modeling_human_lrm.py` 出发，结合以下配置和实际调用模块，整理 LHM-MINI 与 LHM-500M 的模型结构、输入输出张量、SMPL-X query points、Body/Head MMDiT block、Gaussian 属性预测、渲染结果和模型规模。

- `configs/inference/human-lrm-mini.yaml`
- `configs/inference/human-lrm-500M.yaml`
- `LHM/models/modeling_human_lrm.py`
- `LHM/models/transformer.py`
- `LHM/models/transformer_dit.py`
- `LHM/models/rendering/gs_renderer.py`
- `LHM/models/rendering/smpl_x_voxel_dense_sampling.py`

> 说明：本文以当前 inference YAML 和实际构建代码为准。仓库中的 `modelcard.md` 在部分 encoder 名称上与 YAML 不一致。例如，`modelcard.md` 描述 DINOv2 ViT-S，但 MINI 和 500M YAML 均配置为 `dinov2_vitl14_reg`。

## 1. 总体模型结构

LHM 的核心任务不是回归单一 SMPL-X mesh，而是从一张人体图像生成一组可以由 SMPL-X 驱动的 3D Gaussians：

```text
全身参考图像 ──> Sapiens body encoder ──> body image tokens
头部参考图像 ──> FaceSR + DINOv2 ──────> head image tokens

SMPL-X 规范姿态表面采样点 ──> PointEmbed ──> point tokens

point tokens + body/head image tokens
        └──> 多层 Body/Head MMDiT
                └──> 每个 query point 的 latent feature
                        └──> Gaussian 属性预测
                                └──> SMPL-X LBS 驱动到目标姿态
                                        └──> 3D Gaussian Rasterization
                                                └──> RGB / Mask / Depth
```

默认具体模型类为：

```text
ModelHumanLRMSapdinoBodyHeadSD3_5
```

## 2. MINI 与 500M 配置总览

| 配置项 | LHM-MINI | LHM-500M |
|---|---:|---:|
| 模型类 | `ModelHumanLRMSapdinoBodyHeadSD3_5` | 相同 |
| Transformer 类型 | `sd3_mm_bh_cond` | 相同 |
| BodyHead Block 数 | 2 | 5 |
| 每个 BodyHead Block 内部 MMDiT 数 | 2 | 2 |
| MMDiT 总数 | 4 | 10 |
| Transformer channel | 1024 | 1024 |
| Attention heads | 16 | 16 |
| 每个 attention head channel | 64 | 64 |
| Query points / Gaussians | 20,000 | 40,000 |
| Body points | 15,000 | 30,000 |
| Head points | 5,000 | 10,000 |
| Body image tokens | 4096 | 4096 |
| Head image tokens | 1024 | 1024 |
| 图像条件 token 总数 | 5120 | 5120 |
| Gaussian decoder MLP channel | 512 | 512 |
| Gaussian decoder hidden layers | 2 | 2 |
| 训练 source image resolution | 512 | 1024 |
| 训练 render resolution | 384 | 512 |
| 训练侧视角数量 | 7 | 5 |

MINI 和 500M 的主要结构差异只有：

1. BodyHead MMDiT 深度从 2 层增加到 5 层。
2. Query points / Gaussians 数量从 20K 增加到 40K。
3. 训练输入和渲染分辨率不同。

Query points 数量不会明显增加网络权重参数量，但会显著增加 Transformer、SMPL-X skinning 和 Gaussian rasterization 的计算量与显存占用。

## 3. 模型输入

训练 `forward()` 的主要输入为：

```text
image:             [B, N_ref, 3, H, W]
source_head_rgbs:  [B, N_ref, 3, H_head, W_head]
source_c2ws:       [B, N_ref, 4, 4]
source_intrs:      [B, N_ref, 4, 4]

render_c2ws:       [B, N_view, 4, 4]
render_intrs:      [B, N_view, 4, 4]
render_bg_colors:  [B, N_view, 3]
smplx_params:      SMPL-X 形状与目标姿态参数
```

其中：

- `B` 表示 batch 中独立人物样本的数量，通常也对应参考图片数量。
- `N_ref` 是参考视角数量，但当前代码只使用 `image[:, 0]` 和 `source_head_rgbs[:, 0]`。
- `N_view` 是目标渲染视角或目标动作帧数量，与 batch size 不同。
- `source_c2ws` 和 `source_intrs` 虽然传入模型，但当前实现没有使用它们生成 latent features。

典型 `smplx_params`：

```text
betas:       [B, 10]
root_pose:   [B, N_view, 3]
body_pose:   [B, N_view, 21, 3]
jaw_pose:    [B, N_view, 3]
leye_pose:   [B, N_view, 3]
reye_pose:   [B, N_view, 3]
lhand_pose:  [B, N_view, 15, 3]
rhand_pose:  [B, N_view, 15, 3]
expr:        [B, N_view, 100]
trans:       [B, N_view, 3]
```

## 4. Encoder 与图像条件 Token

### 4.1 Sapiens Body Encoder

全身参考图像由冻结的 Sapiens-1B 编码。Sapiens wrapper 会将图像补成正方形并缩放到 `1024 x 1024`。

Sapiens patch size 为 16：

```text
1024 / 16 = 64
64 x 64 = 4096 body tokens
```

输出：

```text
body_feats: [B, 4096, 1536]
```

### 4.2 DINOv2 Head Encoder

头部图像可先经过 FaceSR，再由 DINOv2 Fusion Encoder 编码。DINO wrapper 将输入缩放到 `448 x 448`，patch size 为 14：

```text
448 / 14 = 32
32 x 32 = 1024 head tokens
```

DINO 输出：

```text
head_feats: [B, 1024, 1024]
```

随后在 channel 维补零到 1536：

```text
head_feats_padded: [B, 1024, 1536]
```

### 4.3 图像条件合并

```text
image_feats = concat(body_feats, head_feats_padded)
            = [B, 5120, 1536]
```

进入 Transformer 前通过：

```text
Linear(1536 -> 1024)
```

得到：

```text
encoder_hidden_states: [B, 5120, 1024]
```

## 5. Query Points 的生成

### 5.1 Query Points 的含义

`query_points` 是 SMPL-X 规范姿态表面上的三维采样点：

```text
query_points: [B, N, 3]
```

每个 query point 最终对应一个 Gaussian。它不是最终 Gaussian 中心，而是 Gaussian 的人体表面锚点：

```text
canonical_gaussian_center = query_point + predicted_offset
```

### 5.2 表面采样

默认配置：

```text
latent_query_points_type = e2e_smplx_sub1
smplx_type = smplx_2
body_face_ratio = 3
```

模型首先从 SMPL-X template mesh 中分离 body faces 和 head faces，然后调用：

```python
trimesh.sample.sample_surface(mesh, count)
```

该函数按三角形面积采样三角面，并在三角形内部通过随机重心坐标获得表面点。因此采样是近似表面积均匀采样，而不是均匀选择 mesh 顶点，也不是在包围盒内部采样。

点按如下顺序拼接：

```text
dense_pts = concat(body_points, head_points)
```

所以：

```text
前 75% query points 是 body points
后 25% query points 是 head points
```

MINI：

```text
body points: 15,000
head points:  5,000
total:       20,000
```

500M：

```text
body points: 30,000
head points: 10,000
total:       40,000
```

采样结果会缓存到：

```text
pretrained_models/dense_sample_points/{cano_pose_type}_{dense_sample_points}.ply
```

### 5.3 绑定 SMPL-X 属性

采样点不是原始 SMPL-X 顶点，因此代码通过 KNN 为每个采样点找到最近的 template vertex，并继承：

```text
LBS skinning weights
pose directions
shape directions
expression directions
face / hand / upper-body / lower-body 标签
```

这里使用最近邻属性继承，而不是根据采样点所在三角形做重心插值。

### 5.4 规范姿态

`get_neutral_pose_human()` 从 SMPL-X template/zero-pose 出发，将 dense points 通过预定义 neutral body pose 和 LBS 变换到项目的 canonical pose。

默认：

```text
cano_pose_type = 1
```

对应 `neutral_body_pose[0]` 和 `[1]`，即去除 Pelvis 后的：

```text
L_Hip: +pi/9
R_Hip: -pi/9
```

因此调整的是左右髋关节和双腿张开角度，不是肩膀。

`get_query_points()` 当前使用 `use_id_info=False`，因此 Transformer 输入使用统一模板形状下的规范姿态 query points，而不是每个人根据 `betas` 生成不同的初始 query point cloud。

### 5.5 Query Points 是否归一化

进入 Transformer 前，query point 的原始 xyz 坐标没有进行 AABB 或单位球归一化。

它们直接经过 Fourier PointEmbed：

```text
xyz
 -> 多频率 sin/cos
 -> concat 原始 xyz
 -> Linear
 -> LayerNorm
```

因此：

```text
原始 xyz 坐标没有归一化
PointEmbed 输出 feature 经过 LayerNorm
```

输出：

```text
MINI:  [B, 20000, 1024]
500M:  [B, 40000, 1024]
```

## 6. Channel 变化总览

Transformer 主干内部的 point token 和 image token channel 保持为 1024，但整个模型并非所有 channel 都是 1024。

| 阶段 | Channel / 维度 |
|---|---:|
| Query point xyz | 3 |
| PointEmbed 输出 | 1024 |
| Sapiens body token | 1536 |
| DINO head token 原始输出 | 1024 |
| 补零后的 head token | 1536 |
| 合并图像条件 token | 1536 |
| `linear_cond_proj` 后 image token | 1024 |
| MMDiT point/image token | 1024 |
| Attention heads | 16 |
| 每个 attention head | 64 |
| MMDiT FFN 中间 channel | 通常 4096 |
| 完整 `temb` | 2048 |
| Body / head `temb` | 各 1024 |
| Transformer 输出 latent point | 1024 |
| Gaussian decoder MLP | `1024 -> 512 -> 512 -> 1024` |
| Gaussian 属性输出 | 3 / 3 / 4 / 1 / 3 |

每个 BodyHead MMDiT Block 的主干输入输出 channel 不变：

```text
point tokens: [B, N, 1024] -> [B, N, 1024]
image tokens: [B, 5120, 1024] -> [B, 5120, 1024]
```

## 7. `temb` 的生成与作用

这里的 `temb` 在命名上继承 SD3 timestep embedding，但在 LHM 中实际来自 Sapiens body features。

生成过程：

```text
body_feats: [B, 4096, 1536]
     -> mean over 4096 tokens
[B, 1536]
     -> Linear(1536 -> 768)
     -> SiLU
     -> Linear(768 -> 2048)
[B, 2048]
```

每个 BodyHead Block 中拆分为：

```text
body_temb: [B, 1024]
head_temb: [B, 1024]
```

`temb` 不是作为额外 token 参与 attention，而是通过 `AdaLayerNormZero` 生成：

```text
gate_msa
shift_mlp
scale_mlp
gate_mlp
```

并调制 attention 与 FFN：

```text
hidden += gate_msa * attention_output

normalized_hidden =
    LayerNorm(hidden) * (1 + scale_mlp) + shift_mlp

hidden += gate_mlp * FFN(normalized_hidden)
```

因此 `temb` 用于根据输入人物的全局 body 特征动态控制每个 MMDiT block 的更新。

## 8. BodyHead MMDiT Block

一个 `SD3BodyHeadMMJointTransformerBlock` 包含：

```text
1 个 Head SD3MMJointTransformerBlock
1 个 Body SD3MMJointTransformerBlock
```

### 8.1 Block 总输入

MINI：

```text
point tokens: [B, 20000, 1024]
image tokens: [B,  5120, 1024]
temb:         [B,  2048]
```

500M：

```text
point tokens: [B, 40000, 1024]
image tokens: [B,  5120, 1024]
temb:         [B,  2048]
```

Block 首先拆分：

```text
point tokens:
    前 75% -> body points
    后 25% -> head points

image tokens:
    前 4096 -> Sapiens body tokens
    后 1024 -> DINO head tokens

temb:
    前 1024 -> body_temb
    后 1024 -> head_temb
```

### 8.2 Head MMDiT

MINI 输入输出：

```text
head point input:  [B, 5000, 1024]
head image input:  [B, 1024, 1024]
head temb:         [B, 1024]

head point output: [B, 5000, 1024]
head image output: [B, 1024, 1024]
```

500M 输入输出：

```text
head point input:  [B, 10000, 1024]
head image input:  [B,  1024, 1024]
head temb:         [B,  1024]

head point output: [B, 10000, 1024]
head image output: [B,  1024, 1024]
```

Joint Attention 会将 point 与 image 的 Q/K/V 在序列维拼接。

Joint Attention 总序列长度：

```text
MINI:  5000 + 1024 = 6024
500M: 10000 + 1024 = 11024
```

多头内部张量：

```text
MINI:  [B, 16, 6024, 64]
500M:  [B, 16, 11024, 64]
```

### 8.3 Body MMDiT

Head MMDiT 更新后：

```text
all_points = concat(original_body_points, updated_head_points)
```

所以 Body MMDiT 实际处理全部人体点，而不只是 body points。

MINI 输入输出：

```text
all point input:   [B, 20000, 1024]
body image input:  [B,  4096, 1024]
body temb:         [B,  1024]

all point output:  [B, 20000, 1024]
body image output: [B,  4096, 1024]
```

500M 输入输出：

```text
all point input:   [B, 40000, 1024]
body image input:  [B,  4096, 1024]
body temb:         [B,  1024]

all point output:  [B, 40000, 1024]
body image output: [B,  4096, 1024]
```

Joint Attention 总序列长度：

```text
MINI:  20000 + 4096 = 24096
500M: 40000 + 4096 = 44096
```

多头内部张量：

```text
MINI:  [B, 16, 24096, 64]
500M:  [B, 16, 44096, 64]
```

### 8.4 普通 Block 与最后一个 Block

普通 BodyHead Block 会同时更新：

```text
point tokens
body image tokens
head image tokens
```

最后一个 BodyHead Block 设置：

```text
context_pre_only = True
```

最后一层只保留 point tokens，不再返回 image context tokens。

最终 Transformer 输出：

```text
MINI latent_points: [B, 20000, 1024]
500M latent_points: [B, 40000, 1024]
```

## 9. MINI 与 500M 的 Block 数量

### 9.1 LHM-MINI

```text
BodyHead Block 1:
    Head MMDiT
    Body MMDiT
    更新 point tokens 与 image tokens

BodyHead Block 2:
    Head MMDiT
    Body MMDiT
    最后一个 context_pre_only block
    只保留 point tokens
```

总计：

```text
2 个 BodyHead Blocks
4 个 SD3MMJointTransformerBlocks
```

### 9.2 LHM-500M

```text
BodyHead Block 1-4:
    Head MMDiT
    Body MMDiT
    更新 point tokens 与 image tokens

BodyHead Block 5:
    Head MMDiT
    Body MMDiT
    最后一个 context_pre_only block
    只保留 point tokens
```

总计：

```text
5 个 BodyHead Blocks
10 个 SD3MMJointTransformerBlocks
```

## 10. 单个 MMDiT 的内部结构

普通 `SD3MMJointTransformerBlock` 主要包含：

```text
Point AdaLayerNormZero
Image AdaLayerNormZero

Point Q/K/V projection
Image Q/K/V projection
Joint Attention
Point output projection
Image output projection

Point FeedForward
Image FeedForward
```

主干维度：

```text
D = 1024
attention heads = 16
head_dim = 64
FFN hidden dim approximately 4096
```

一个普通 MMDiT 的参数量约为：

```text
37.8M
```

一个普通 BodyHead Block 包含两个普通 MMDiT，参数量约为：

```text
75.6M
```

最后一个 BodyHead Block 不再保留 image context 输出，因此参数量略少，约为：

```text
46M - 48M
```

## 11. Gaussian 属性预测

配置使用：

```text
latent_query_points_type = e2e_smplx_sub1
```

因此：

```text
skip_decoder = True
```

Transformer 输出与 query points 一一对应，不再经过额外的 cross-attention decoder。

每个 latent point feature 经过：

```text
MLP: 1024 -> 512 -> 512 -> 1024
```

再由独立 linear heads 预测：

```text
offset_xyz: 1024 -> 3
scaling:    1024 -> 3
rotation:   1024 -> 4
opacity:    1024 -> 1
RGB:        1024 -> 3
```

单个 batch 样本内部的 `GaussianAppOutput` 不保留 batch 维：

```text
offset_xyz: [N, 3]
scaling:    [N, 3]
rotation:   [N, 4]
opacity:    [N, 1]
RGB:        [N, 1, 3]
```

多个 batch 样本通过长度为 `B` 的 Python list 保存。模型返回时，部分属性重新 stack：

```text
offset_output:  [B, N, 3]
scaling_output: [B, N, 3]
```

### 11.1 Gaussian 属性初始化

项目没有按采样点法向量初始化 rotation，也没有按邻域点距离初始化 scaling。

初始化为：

```text
offset_xyz = 0
Gaussian center = query_point

scaling = exp(-5) approximately 0.00674
所有轴相同，因此初始 Gaussian 为球形

rotation = identity quaternion [1, 0, 0, 0]
opacity = 0.1
```

由于初始 scaling 是各向同性球体，初始 rotation 对渲染没有影响。

更几何化的可选方案是：

```text
base position = query point
base rotation = surface-normal-aligned rotation
base scaling = KNN-distance-based anisotropic scaling

final attributes = base geometry + learned residual
```

但当前项目没有采用该方案。

## 12. SMPL-X 驱动与渲染输出

Canonical Gaussian 中心：

```text
mean_3d = query_points + offset_xyz
```

随后通过 SMPL-X LBS / voxel skinning 驱动至每个目标姿态。Gaussian rotation 也会乘以目标姿态对应的局部刚性旋转。

渲染输出：

```text
comp_rgb:   [B, N_view, 3, H, W]
comp_mask:  [B, N_view, 1, H, W]
comp_depth: [B, N_view, 1, H, W]
3dgs:       每个视角的 GaussianModel
```

同一个人物的多个目标视角共享同一套 canonical Gaussian 属性，只是使用不同 SMPL-X pose 和相机进行变形、渲染。

## 13. 核心参数量对比

以下参数量根据当前代码结构估算，不包含外部 Sapiens、FaceSR、GFPGAN 和 ArcFace。由于运行环境缺少完整依赖与预训练权重，数值应视为结构级近似值。

| 模块 | LHM-MINI | LHM-500M |
|---|---:|---:|
| PointEmbed | 约 0.055M | 约 0.055M |
| Image condition projection | 约 1.574M | 约 1.574M |
| Motion / `temb` MLP | 约 2.755M | 约 2.755M |
| BodyHead MMDiT Blocks | 约 124M | 约 351M |
| Gaussian decoder MLP | 约 1.313M | 约 1.313M |
| Gaussian attribute heads | 约 0.014M | 约 0.014M |
| 核心重建网络合计 | 约 130M | 约 356M |

Query points 从 20K 增加到 40K 基本不会增加网络权重参数量，因为 Transformer 与 Gaussian heads 对所有点共享权重。

### 13.1 “500M” 命名说明

当前 MINI 和 500M YAML 均配置：

```text
DINOv2 ViT-L/14 register
Sapiens-1B
```

DINOv2 ViT-L 本身约 300M 参数，Sapiens-1B 约 1B 参数且被冻结。模型还可能加载 ArcFace、RealESRGAN 和 GFPGAN。

因此如果把 Python 对象中所有冻结与外部组件都计入，总参数量明显超过 500M。`LHM-500M` 更适合理解为作者对主要生成模型规模的命名，而不是当前对象内所有参数严格求和后的数值。

## 14. Attention 计算量对比

Attention 的主要计算量近似与联合序列长度平方成正比。

### 14.1 每个 BodyHead Block 的 token-pair 数

MINI：

```text
Head: 6024^2
Body: 24096^2
合计约 0.617B token pairs / block
```

500M：

```text
Head: 11024^2
Body: 44096^2
合计约 2.066B token pairs / block
```

单个 500M BodyHead Block 的 attention 计算规模约为 MINI 单个 Block 的：

```text
2.066 / 0.617 approximately 3.35 times
```

考虑 block 数：

```text
MINI:  2 x 0.617B
500M: 5 x 2.066B
```

总体 attention 计算规模比例约为：

```text
500M / MINI approximately 8.37 times
```

实际实现使用 memory-efficient attention，因此不会显式保存完整 attention matrix，但序列长度和 block 数仍然显著影响运行时间与显存。

## 15. 最终对比总结

### LHM-MINI

```text
2 个 BodyHead Blocks
4 个内部 MMDiT
20K query points / Gaussians
Transformer 输出: [B, 20000, 1024]
核心重建网络约 130M 参数
更低的输入与渲染分辨率
```

### LHM-500M

```text
5 个 BodyHead Blocks
10 个内部 MMDiT
40K query points / Gaussians
Transformer 输出: [B, 40000, 1024]
核心重建网络约 356M 参数
更高的输入与渲染分辨率
总体 attention 计算规模约为 MINI 的 8.4 倍
```

两者的主干 channel 都保持为 1024。500M 的主要优势来自更深的 Body/Head MMDiT 和更密集的 Gaussian 表示，而不是更宽的 Transformer。
