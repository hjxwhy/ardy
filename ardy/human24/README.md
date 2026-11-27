# Human24 FSQVAE

输入和重建按顺序包含 23 个关节的骨盆相对位置、24 个关节相对窗口首帧的全局旋转（6D）、四个足部接触标记和骨盆高度，共 218 维。位置和高度以米为单位，输出的接触通道是概率，可用 0.5 二值化。每 4 帧产生一个 128 维 FSQ token，每个维度有 64 档。整个 218 维窗口都经过编码器，解码仅需要这些 token。

在仓库根目录可运行下面的示例。

~~~python
import json
import torch
from ardy.human24.data import DATA, prepare_clip, make_chunk
from ardy.human24.model import Human24FSQVAE

parents = json.loads((DATA / "skeleton.json").read_text())["parents"]
clip = prepare_clip(DATA / "230315/egypt_dance_R_003__A275.npz", parents)
x = torch.from_numpy(make_chunk(clip, start=32, length=64)).unsqueeze(0)
model = Human24FSQVAE.from_checkpoint("outputs/human24_fsq/best.pt")
with torch.inference_mode():
    codes = model.encode(x)                   # [1, 16, 128]
    reconstruction = model.decode(codes, 64)  # [1, 64, 218]
~~~

恢复绝对全局旋转需要另存窗口首帧的 24 个全局旋转，恢复世界位置需要另存骨盆 XZ 轨迹。

## 复现训练

第一条命令写入新的本地缓存目录，需要约 32 GB 空间。训练按路径哈希留出约 2% 的动作，同一原版和镜像共享划分。对每个训练片段，先在整段上计算全局旋转和接触，再将可选起点分四段并各抽一个起点。验证集每个片段固定一个起点。短于 64 帧的动作在尾部复制末帧，损失和验证指标只统计有效帧。缓存中的 `*_clips.txt` 与 `*_spans.npy` 分别记录片段清单和窗口的起点、有效长度。

~~~bash
OPENBLAS_NUM_THREADS=1 python -m ardy.human24.data \
    --data /mnt/data/datas/bones-seed/human24 --cache /tmp/human24_windows --workers 32
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 -m ardy.human24.train \
    --cache /tmp/human24_windows --output outputs/human24_fsq --epochs 4 --batch 128
~~~

训练器会在每张卡上加载窗口数据。本数据集的 4 窗口设置仅输入张量就需要每卡约 30 GiB 显存。每个 epoch 遍历全部训练窗口，最后一个 batch 用少量重复窗口补齐。位置、旋转、骨盆高度分别计算标准化特征的 Smooth L1 损失，接触通道计算二元交叉熵，四项等权。标准化统计量仅从训练窗口估计并保存在权重中。验证记录含骨盆相对关节位置平均欧氏误差、相对旋转平均测地角误差、骨盆高度绝对误差和足接触 micro-F1。

`best.pt` 按验证损失选取，`last.pt` 保存最近一个完整 epoch 和优化器状态。相同数据及参数的训练中断后，可在同一训练命令末尾添加 `--resume`。输出目录下的 `run.json` 和 `metrics.jsonl` 分别记录训练配置和逐轮验证结果。

## 从已有权重继续全帧训练

`--full-train` 在每段训练动作上依次取 64 帧窗口，最后一个窗口对齐动作末帧。这样每个训练帧每轮至少出现一次。验证窗口沿用原来的划分、起点和随机种子。缓存中的 `train_clip_offsets.npy` 给出每段动作在窗口表中的起止行，`train_clip_frames.npy` 给出各段真实帧数，可与 `train_spans.npy` 联合核验覆盖率。

~~~bash
OPENBLAS_NUM_THREADS=1 python -m ardy.human24.data --full-train \
    --data /mnt/data/datas/bones-seed/human24 --cache /tmp/human24_allframes --workers 32
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=4 -m ardy.human24.train \
    --cache /tmp/human24_allframes --output outputs/human24_fsq_full \
    --init-from outputs/human24_fsq/best.pt --lr 0.00015 --epochs 4 --batch 128
~~~

此次全帧缓存需要约 111 GB 本地空间。四张 GPU 各加载一个训练数据分片并各自打乱窗口顺序。新训练从给定权重及其标准化统计量开始，使用新的 AdamW 优化器和学习率计划。加载示例中的权重路径可改为 `outputs/human24_fsq_full/best.pt`。

## 在训练集群运行 30 Hz、每卡 128 的 Human24 模型

`scripts/train_human24_paper.sh` 支持从随机权重训练 Human24 tokenizer，并可从单卡断点切换到多机多卡 DDP。使用 Linux、Bash、flock、Python 3.10+、支持 CUDA 的 PyTorch 2.4+、NumPy 和 SciPy。每个进程绑定一张支持 BF16 的 GPU；当前配置适合每卡 80 GB 的节点。训练进程直接从共享挂载的原始 NPZ 读取窗口，内存中预取下一批。将下面命令放入集群作业的启动命令或批处理脚本中；如果挂载路径不同，请设置对应路径。

~~~bash
export DATA_DIR=/mnt/data/datas/bones-seed/human24
export OUTPUT_DIR=/mnt/data/users/jensen/code/code_latest/ardy/outputs/human24_fsq_paper_recheck
export DATA_WORKERS=4
export NPROC_PER_NODE=8
bash scripts/train_human24_paper.sh
~~~

启动时 rank 0 扫描源文件头，广播路径和帧数索引。每个 rank 使用 `DATA_WORKERS` 个线程读取所分配的窗口；已有本次训练的 `last.pt` 时，恢复模型、优化器、更新步数和数据游标。使用 PAI DLC 的 PyTorch 作业时，每台机器执行一次相同的 Bash 命令。脚本从平台读取节点数 `WORLD_SIZE`、节点编号 `RANK`、每节点卡数 `NPROC_PER_NODE` 和 `MASTER_ADDR/MASTER_PORT`，用 torchrun 为每张卡启动一个 DDP 进程。外部 torchrun 已设置 `LOCAL_RANK` 时，脚本使用已有的 rank 环境初始化进程组。其他启动平台可以显式设置 `NNODES`、`NODE_RANK`、`NPROC_PER_NODE`、`MASTER_ADDR`、`MASTER_PORT`。未指定 `NPROC_PER_NODE` 时默认每台启动 8 个进程。所有节点共享原始 NPZ 路径和输出目录。

每个 rank 每次取 128 个窗口，同步梯度后更新一次。每台使用 8 卡时，单台的全局 batch 为 1,024，两台为 2,048。首次在新的 30 Hz 输出目录中从随机权重开始，后续可从该目录的 30 Hz 断点续训。验证、日志和断点由全局 rank 0 写入。启动日志中的 `world_size` 是总进程数，`per_rank_batch_min/max` 是每卡窗口数，`resume_step` 是成功加载的断点步数。迁移旧作业前应先结束其所有进程，再向同一输出目录提交新作业，以免旧进程继续写入。作业超时后可用相同命令重新提交。

| 设置 | 本训练脚本 |
|---|---|
| 模型 | 编码器和解码器各 8 层因果 Transformer，宽 512，4 个头，FFN 1024，GELU，post-norm，零 dropout |
| FSQ | 每 4 帧一个 128 维 token，每个标量 64 档 |
| 优化 | AdamAtan2，400 万次更新，每卡 batch 128，峰值学习率 2e-5，前 10,000 步线性 warmup，其后余弦衰减 |
| 损失 | 标准化位置、旋转和骨盆高度各一项 Smooth L1，加足接触 BCE，再加 0.01 倍滑步损失 |
| 窗口 | 将 120 Hz 源数据每 4 帧采样到 30 Hz，以相邻采样帧的世界足速判定接触，再选取窗口并计算首帧相对旋转 |

每轮随机分割训练片段，生成 1–10 秒窗口，遍历所有降采样后的训练帧。训练集中有 36 个源片段降采样后不足 30 帧，补至 1 秒后用掩码排除填充帧。验证沿用原有独立留出的动作文件，并使用固定的 30 Hz 窗口。与 120 Hz 的 64 帧验证相比，时间跨度和窗口起点均有变化。

模型的 218 维接口与原始 ARDY 的 root/body 条件接口不同。解码器只接收 FSQ 向量。滑步损失使用预测的足部接触概率对世界足速度加权。足速度由相邻帧重建的骨盆相对足位置、预测骨盆高度，以及仅在训练和验证损失中使用的真实骨盆水平位移计算。推理不需要水平轨迹。

作者没有公开 FSQ 重建损失的逐通道权重、优化器的具体 betas 或余弦终点。此实现采用上述分组重建损失、AdamAtan2 的 betas=(0.9,0.99)、a=1.27、b=1.0、零权重衰减，余弦终点为 0。[论文训练细节](https://arxiv.org/html/2607.08741v1#S3.SS5) · [官方 Core 模型配置](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40/raw/main/config.yaml)

首个断点和验证结果在第 100 步写入；之后每 1,000 步更新 `last.pt`，每 5,000 步验证并按验证总损失选择 `best.pt`，第 4,000,000 步另存 `final.pt`。学习率、速度、损失和验证指标记录在 `metrics.jsonl`。推理可用上文的 `Human24FSQVAE.from_checkpoint` 加载模型，读取推理输入时使用 `prepare_clip(path, parents, stride=4, expected_fps=120)`，再调用 `make_chunk`。这里的窗口起点和长度按 30 Hz 计数。

训练器把接触速度的采样率写入 `run.json` 的 `contact_velocity_fps`。新实验（包括用 `INIT_FROM` 启动的新目录）使用 30 Hz 标签。同目录续训会沿用该实验的采样率。缺少此字段的旧 30 Hz 实验使用 120 Hz 标签，复现这些实验的推理输入时传入 `prepare_clip(..., stride=4, source_rate_contacts=True)`。

### 归档实验

原始标量 FSQ 是训练脚本的唯一模式；新一轮默认写入 `outputs/human24_fsq_paper_recheck`。新训练按官方实现将投影结果与 FP32 正弦位置编码相加。旧断点按保存时的精度加载。PCA 粗档实验的推理和续训权重、源码、配置和校验和保存在 `backups/human24_motion_grid_20261126T035523Z/`，其中的 `source.tar.gz` 含当时的启动脚本。
