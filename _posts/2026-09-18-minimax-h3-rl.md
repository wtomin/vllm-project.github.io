---
layout: post
title: "使用 VeRL-Omni 与 vLLM-Omni 对 MiniMax-H3 进行音视频强化学习后训练"
author: "VeRL-Omni Team"
summary: "介绍如何使用 DiffusionNFT、VeRL-Omni 与 vLLM-Omni 跑通 MiniMax-H3 的 T2VA 和 FL2VA 在线强化学习后训练，并解决 rollout 性能与训推一致性问题。"
image: /assets/logos/vllm-logo-text-light.png
tags:
  - multimodal
  - rlhf
  - ecosystem
  - performance
  - vllm-omni
published: true
---

## 引言

MiniMax-H3 是一个通用的多模态生成模型，可接收文本、图像、视频、音频等多种输入，直接输出带原生立体声音轨的视频，支持最长 15 秒、最高 2K 分辨率、24 FPS、32 kHz 立体声。权重已在 Hugging Face 开源。在此基础上通过 RL 后训练，可进一步提升其在垂类领域的能力。

不过，对音视频联合生成模型做 RL 后训练，真正麻烦的不是挑算法或设计 reward，而是如何确认整条链路确实跑对了。脚本跑得起来、loss 有数、reward 曲线在动，这些都不能说明训练方向没有问题。**在联合音视频场景里，任何一环悄然出错，训练曲线仍可能稳步上升——这才是最棘手的地方。**

本文记录 MiniMax-H3 在 VeRL-Omni 和 vLLM-Omni 中的后训练实践。我们以 DiffusionNFT 算法为例，跑通了两条在线 RL 训练闭环：一条是 T2VA（text-to-audio-video），另一条是 FL2VA（first/last-frame-to-audio-video）。具体分工如下：vLLM-Omni 负责 rollout，即采样音视频；Diffusers / FSDP2 负责训练 actor；VeRL-Omni 将数据、reward、NFT loss、LoRA 旧策略同步等环节串成完整链路。

对联合音视频的 RL 后训练来说，rollout 环节集中了两大难点。第一，音视频生成非常消耗算力。rollout 在端到端耗时中占据大头，其性能基本决定了训练效率。第二，训练引擎（Diffusers / FSDP2）和推理引擎（vLLM-Omni）存在差异，包括权重切分、LoRA 命名、timestep 的数值范围等。因此，必须单独处理 rollout 的关键配置和调度，才能保证训练侧和采样侧使用同一个策略。本文围绕这两条主线展开：如何利用 vLLM-Omni 的高吞吐 rollout 和参数同步机制，在 VeRL-Omni 中真正跑通 MiniMax-H3 的联合音视频在线 RL 闭环。

## 1. 背景：MiniMax-H3 和 DiffusionNFT

### 1.1 MiniMax-H3：联合音视频生成

MiniMax-H3 生成的并非单独一段视频张量。它同时产出 video latent 和 audio latent，二者最终一起解码成带音轨的视频。这个特性为 RL 训练提出了两项要求：

- reward 必须同时评估画面和声音，不能只关注一个模态；
- 训练框架必须保证 audio 和 video 在 rollout、reward、数据落盘和训练 batch 之间都不会悄然丢失。

H3 的 DiT 主干还有几处与通用约定不同：它经过 CFG distillation，推理时不需要 negative prompt；timestep 使用 data fraction，而不是流匹配中常见的 sigma；velocity 的符号方向也与通用约定相反。这些细节在接入训练框架时都可能成为隐患，后文会具体说明。

### 1.2 DiffusionNFT 算法原理

DiffusionNFT 的全称是 Diffusion Negative-aware FineTuning，论文标题为 **DiffusionNFT: Online Diffusion Reinforcement with Forward Process**。它是一种面向 diffusion / flow-matching 模型的在线 RL 方法。它不在反向采样链上估计 policy gradient，而是将 reward 信号注入前向扩散过程的监督式 flow-matching 目标。rollout 端只保留最终生成的 clean latent、prompt embedding 和训练 timestep；训练端重新执行一次前向加噪，将 reward 转换成 reward probability，使高 reward 样本更接近正向目标，同时反向拉住低 reward 样本。

第一阶段先接入 DiffusionNFT，是因为接入 H3 时最需要优先验证以下三项基础事实：

- vLLM-Omni 的 rollout 是否真的**同时**生成了 video 和 audio；
- CLAP / ImageBind 的 reward 是否真的拿到了完整的音视频；
- actor 更新 LoRA 后，是否真的同步回了下一轮 rollout。

DiffusionNFT 不需要在 rollout 端记录每一步 transition 的 log-prob，链路更短，因此可以逐项验证以上问题。后来我们又接入了 FL2VA。T2VA 和 FL2VA 目前共用 rollout、reward、actor 和 LoRA 同步这条主干；二者的差别只在条件帧输入和训练 loss 的掩码范围。

下图汇总了两条任务路径：上半部分是 T2VA / FL2VA 的条件输入和 H3 rollout；中间是音画 reward 和训练数据格式；下半部分是 DiffusionNFT 的前向过程优化，以及 old policy 的刷新。阅读时需要区分两条数据流：**用于评分的解码视频/音频流向 CLAP 与 ImageBind；用于 actor 更新的 clean latents、timestep 与条件元数据流向 FSDP2。**

![](/assets/figures/2026-09-18-minimax-h3-rl/image.png)

上图展示了系统分层和主要数据流。目前 DiffusionNFT 默认使用**全局标准差 reward 归一化**。rollout policy 的更新细节见第 4.2 节。

## 2. 为什么 rollout 是音视频 RL 后训练的关键

对音视频联合生成模型做 RL 后训练，rollout 环节集中了两个风险：**计算量**和**训推一致性**。

### 2.1 计算量：音视频生成让 rollout 成为时间瓶颈

与文本或纯图像 RL 不同，联合音视频 rollout 每生成一个样本，都需要完整执行 H3 的 denoise loop，还要解码视频帧并合成 32 kHz 立体声音轨。单条样本的生成成本远高于文本生成。DiffusionNFT 还要求对同一个 prompt 采集多条 rollout（默认 `ROLLOUT_N=16`），以构造组内 reward 差异，这进一步放大了 rollout 的总计算量。

第 7.3 节的耗时分解显示：将端到端耗时拆分为 rollout、reward、actor update 和 checkpoint 后，rollout 和 reward 是最大的两个部分（reward 阶段还要同时进行音频解码、视频处理并运行两个 scorer）。**rollout 是主要的生成瓶颈，reward 是联合音视频场景中随之而来的伴生瓶颈；端到端优化必须同时匹配二者的吞吐。**

### 2.2 训推一致性：训练引擎和推理引擎不同

在 H3 的在线 RL 链路中，rollout 由推理引擎 vLLM-Omni 执行，actor 训练由 Diffusers / FSDP2 执行。两个引擎在多处存在差异：

- **权重布局**：训练侧看到的是拆分的注意力投影，即 `to_q / to_k / to_v / to_out.0`；vLLM-Omni 中的 H3 将它们融合成了一块 fused DiT；
- **LoRA 命名**：Diffusers 的 LoRA 模块命名与 vLLM-Omni 的融合结构并非一一对应。缺少映射时，adapter 可以正常注册，但实际命中的层数为 0；
- **timestep / velocity 约定**：H3 的扩散时间步使用 data fraction，而不是通用 flow-matching 的 sigma；velocity 的符号方向也与通用约定相反。训练端如果直接按照通用约定传入，loss 仍然可以计算出数值，但梯度方向可能完全相反。

这些差异有一个共同的危险特征：**不报错**。loss 有数值，adapter 注册成功，曲线也在变化，但梯度方向可能反了，或者更新根本没有生效。因此，rollout 的关键配置和调度——尤其是训练 adapter 到 rollout adapter 的参数同步、fused DiT 映射、target modules 校验和 old policy 刷新——必须单独处理。这也是第 4 节的主题：训推一致性的配置和调度。

## 3. Rollout 性能优化

H3 能够较为顺利地接入 VeRL-Omni，得益于仓库中几个核心模块的清晰解耦：vLLM-Omni 只负责生成，VeRL-Omni 只负责训练编排，H3 adapter 专门吸收模型相关约定，包括如何切分 packed latent、如何换算 timestep、velocity 符号和 LoRA 命名映射。模型特性基本都被封装在 pipeline adapter 中，通用训练循环无需了解 H3 的每个细节。将 rollout 交给 vLLM-Omni，而不是由训练引擎执行，正是因为 rollout 是吞吐瓶颈：vLLM-Omni 的推理优化——连续 batching、fused kernel 和张量并行——能够明显降低单条 rollout 的耗时；而训练引擎（Diffusers / FSDP2）原本就不是为高吞吐生成设计的。

### 3.1 张量并行配置需要在吞吐和同步之间权衡

rollout 的张量并行度同时影响吞吐和训推同步能力，需要在两者之间权衡：

| 任务 | rollout TP | 说明 |
|---|---:|---|
| T2VA | 2 | 默认配置 |
| FL2VA | 4 | 使用 TP=4，为 actor-to-rollout 同步留出显存余量 |

FL2VA 的条件帧协议和条件帧固定逻辑会额外消耗显存，因此将 rollout TP 从 2 提高到 4，为 actor-to-rollout 的 LoRA 同步留出余量。

### 3.2 rollout 与 reward 的吞吐协同

端到端耗时可以拆分为 rollout、reward、actor update 和 checkpoint。在联合音视频训练中，reward 阶段需要同时执行音频解码、视频处理和两个 scorer，因此它和 rollout 一样可能成为瓶颈。优化训练效率时，不能只看 actor 的 GPU 利用率。

将 reward worker 数设为 1，以避免多进程占满第一张卡——这是 reward 和 rollout 共用一张 GPU 时对显存与算力的权衡。rollout 和 reward 的吞吐需要一起规划：reward 过慢时，rollout 采出的样本会堆积，actor 无法及时获得数据；reward 较快而 rollout 较慢时，actor 又会空转。各阶段的实际耗时分解见第 7.3 节。

## 4. 训推一致性：rollout 的关键配置和调度

### 4.1 保留 H3 原生 denoise loop

vLLM-Omni 的 rollout 保留 H3 原生 denoise loop，训练侧不需要重新实现 H3 的采样过程。这是保障训推一致性的第一道防线：如果训练侧自行实现采样，调度器、timestep 或 velocity 约定上的任何差异，都可能使训练侧和采样侧的生成分布悄然分叉。保留原生 loop，相当于将“如何采样”交由同一份代码处理，训练侧只接收结果。

### 4.2 LoRA 参数同步和 old policy 刷新

actor 训练完成后，LoRA 权重必须同步回旧的 rollout policy，否则下一轮 rollout 仍会使用旧策略。同步通过 `policy_state_adapters` 配置实现，同时维护 `default` 和 `old` 两路 adapter：

```bash
actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
algorithm.old_policy_update_interval=2 \
```

`old` policy 的更新并非无条件硬拷贝，而是采用延迟线性衰减调度（`delayed_linear_to_0_999`），每两个训练步刷新一次。该调度让 old policy 平滑跟随训练 policy，避免每步硬拷贝造成分布跳变；同时也能控制训练侧和采样侧策略的偏离幅度——偏离过大时，ref KL 会失效，训推一致性也无从谈起。该同步是训推一致性的核心环节：训练 adapter 在 Diffusers 命名空间中更新，rollout adapter 在 vLLM-Omni 的 fused 命名空间中生效，二者之间必须进行映射（见第 4.3 节）。

### 4.3 Fused DiT 映射和 target modules 校验

训练端看到的是拆分的注意力投影，vLLM-Omni 中的 H3 则将它们融合成了一块。缺少映射时，adapter 可以正常注册，实际命中的层数却为 0——下一轮 rollout 仍会表现为 base model 的行为。

因此，必须显式列出 target modules，不能使用 all-linear：

```bash
actor_rollout_ref.model.lora_rank=64 \
actor_rollout_ref.model.lora_alpha=128 \
actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"
```

**H3 不能直接使用 all-linear 作为 target modules。** 也并非所有训练端 LoRA 都能同步到 rollout 端（例如 fused QKV 必须显式映射）。adapter 会预先校验 target modules，配置错误时直接报错并中止，避免训练悄然偏离。这是训推一致性的最后一道关卡：fused DiT 映射或 LoRA 命名的任何错误，都会在校验阶段暴露，而不是隐藏在训练曲线中。

### 4.4 timestep / velocity 约定的换算

H3 的扩散时间步使用 data fraction（数据占比，即从数据到噪声的插值比例），而不是通用 flow-matching 的 sigma；velocity 的符号方向也与通用约定相反。接入时，只需确保 H3 adapter 按照原生约定完成 timestep 换算，并统一 velocity 符号。否则，loss 仍可能产生数值，但梯度方向会出错。

## 5. 实验设计和配置

实验需要回答的四个问题，都与数据、条件和配置是否发生悄然错配有关。本节汇总任务定义、rollout 返回的训练数据格式、条件帧协议和可复现配置。

### 5.1 实验目标和范围

本文实验不追求 benchmark SOTA 的横向比较。第一阶段的目标更加基础，也更容易被忽视：**我们需要确认 H3 的联合音视频数据、reward 信号和策略更新，在同一条在线 RL 链路中没有悄然错配。**

因此，实验围绕以下四个问题展开：

1. **联合输出是否完整？** rollout 返回的 video 和 audio 是否同时进入数据落盘、reward 和训练数据格式；
2. **reward 是否真正约束音画？** CLAP 是否同时看到了文本和音频，ImageBind 是否看到了同一条样本的音频和视频；
3. **更新是否传回采样策略？** actor 侧的 LoRA 更新后，旧的 rollout policy 是否按照既定调度刷新；
4. **条件帧是否得到正确处理？** FL2VA 中输入的关键帧必须保持为条件，不能被 DiffusionNFT 目标重新优化。

这也决定了结果的解读方式：需要结合 reward 曲线、子 reward、固定样本视频和训练日志进行判断。任何单一指标的上升，都不足以证明生成质量整体提升。

### 5.2 两类任务：T2VA 和 FL2VA

| 任务 | 模型输入 | 条件约束 | 训练中被优化的输出 |
|---|---|---|---|
| **T2VA**（text-to-audio-video） | 文本 prompt | 不使用首帧，也不使用 negative prompt；H3 本身经过 CFG distillation | 全部生成的 video / audio latent |
| **FL2VA**（first/last-frame conditioned text-image-to-audio-video） | 文本 prompt + 一张或两张关键帧 | 首帧、尾帧或首尾帧 | 生成的视频和音频 latent；条件帧 latent 固定 |

T2VA 用于检验文本、运动和音画语义是否对齐。FL2VA 更进一步，检验模型能否在给定关键帧的约束下补全中间运动和音轨，适合角色一致性、镜头衔接和广告素材延展等任务。

两条任务共用 rollout、reward、actor 和 LoRA 同步主干，差别在于条件输入和训练 loss 的掩码范围。FL2VA 的 rollout 使用 vLLM-Omni 官方的首/尾帧协议：**条件图像对应的 latent 保持固定，DiffusionNFT 的前向过程目标只施加在生成的视频和音频 latent 上**——这决定了模型是在学习补全，还是错误地改写关键帧。数据格式还包含条件帧的分段信息和 `frame_indices`，actor 据此固定条件帧 latent。`FRAME_INDICES` 必须与数据转换时的 `frame_mode` 对应，启动时的具体取值见第 6.2 节。

### 5.3 训练数据格式：联合打包和 pipeline 约束

为了确认联合输出完整，rollout 返回的字段必须能够同时支持 actor 更新和音画 reward。rollout 端需要一并返回 DiffusionNFT 训练所需的状态：

```python
rl = {
    "latents_clean": pack_video_audio_rows(video_rows, audio_rows),
    "train_timesteps": train_timesteps,
    "latent_meta": latent_meta,
}
```

rollout 负责“生成并打包”，actor 负责“拆包并训练”；任何字段缺失或错位，都会导致训练端获得错误目标。

`latents_clean` 是视频和音频 latent 拼接后的结果，`latent_meta` 在训练端指导拆分。audio 最容易在链路中段悄然丢失——许多 diffusion / 视频训练框架默认只处理画面，而 H3 输出的是音视频联合结果。**只有 audio 真正送入 CLAP / ImageBind，且导出的视频也包含音轨，reward 链路才算闭合。**

recipe 中的帧数和分辨率也受 pipeline 约束：`NUM_FRAMES=96` 会在 pipeline 中对齐为 107 帧，必须满足 17n+5 边界（24 FPS 下视频时长为 4–15 秒）；采样边长必须为 32 的倍数，否则 H3 pipeline 会悄然向下取整。这些默认值列于第 5.5 节。

### 5.4 数据集和条件图

T2VA 的数据格式是仅包含 prompt 的 parquet。转换器接受两种输入：每行一个 prompt 的纯文本文件，或每条记录带有 prompt 字段的 JSONL；输出训练集和测试集两份 parquet。具体转换命令见第 6.1 节。

FL2VA 使用 DanceGRPO 发布的 ConsisID prompt 列表构造条件图数据：每条 prompt 使用固定 seed 加索引生成一张 FLUX.1-dev 参考图，再使用同一个索引将 prompt 和图像配对，以保证中断恢复、训练测试划分和条件帧都可复现。按照 seed-42 划分后，得到 27,687 条训练样本和 128 条测试样本。条件图的具体生成、JSONL 划分和 parquet 转换命令见 recipe。条件图可以直接复用，缩放由 rollout pipeline 在运行时通过 LANCZOS 处理。

### 5.5 可复现的训练配置

下表列出了当前主分支 recipe 的关键默认值。这些参数已通过端到端验证，可以直接作为复现实验的起点。

| 配置项 | T2VA recipe | FL2VA recipe |
|---|---|---|
| GPU 数 | 8 | 8 |
| rollout TP | 2 | 4 |
| 每个 prompt 的 rollout 数 n | 16 | 16 |
| 训练 / 验证分辨率 | 256x384 / 512x768 | 288x448 / 576x928 |
| 帧数 | 121 | `NUM_FRAMES=96`，由 pipeline 对齐为 107 帧 |
| rollout / validation steps | 10 / 40 | 10 / 40 |
| LoRA | rank 64, alpha 128 | rank 64, alpha 128 |
| Old policy 更新 | 延迟线性衰减，每 2 个训练步刷新 | 同左 |
| reward | CLAP + ImageBind audio-video | CLAP + ImageBind audio-video |

### 5.6 评估指标和检查项

训练过程记录四类信息：

- **优化统计量**：loss、梯度范数（grad norm）、reward probability、reference KL（ref KL）；
- **reward 分解**：CLAP、ImageBind 和 weighted reward；
- **固定样本对比**：使用相同的 prompt、seed 和推理配置，对比 base model 和更新后的策略；
- **系统链路检查**：audio 是否以 32 kHz 送入 CLAP / ImageBind，导出的 MP4 是否包含 AAC 音轨，LoRA 是否命中 rollout 中实际执行的层。

FL2VA 还需要额外检查条件帧的保持情况：首帧身份是否保持、prompt 场景是否成立，以及整段视频内的时间一致性是否达标。

## 6. 训练流程

### 6.1 T2VA 数据准备和启动

首先，将仅包含 prompt 的原始数据转换为 parquet：

```bash
python3 examples/diffusionnft_trainer/minimax_h3/prepare_t2va_data.py \
  --input_dir /path/to/raw_prompts \
  --output_dir /path/to/h3_t2va_data
```

然后启动训练：

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/h3_t2va_data

NUM_GPUS=8 ROLLOUT_TP=2 ROLLOUT_N=16 INFER_STEPS=10 \
TOTAL_TRAINING_STEPS=1000 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh
```

`MODEL_PATH` 指向本地 MiniMax-H3 根目录，其中需要包含 rollout 使用的 `FL2VA/`（vLLM-Omni 检查点目录，T2VA 和 FL2VA 共用），以及训练使用的 `transformer/`。

### 6.2 启动 FL2VA 训练

FL2VA 使用独立的 image-conditioned rollout adapter 和 agent loop，不能直接套用 T2VA 脚本并只更换数据目录。需要确认 `DATA_DIR` 下已经存在 `train.parquet` 和 `test.parquet`，并让 `MODEL_PATH` 指向同时包含 `FL2VA/` 和 `transformer/` 的 MiniMax-H3 根目录。

首帧条件训练的最短启动命令如下：

```bash
export MODEL_PATH=/path/to/MiniMax-H3
export DATA_DIR=/path/to/h3_fl2va

FRAME_INDICES='[0]' \
NUM_GPUS=8 ROLLOUT_TP=4 ROLLOUT_N=16 INFER_STEPS=10 \
TOTAL_TRAINING_STEPS=1000 OUTPUT_DIR=/path/to/output \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

`FRAME_INDICES` 必须与数据转换时的 `frame_mode` 对应：首帧数据使用 `'[0]'`，尾帧数据使用 `'[-1]'`，首尾帧数据使用 `'[0,-1]'`。切换条件模式时，只需重新转换 parquet，再传入对应的 `FRAME_INDICES`：

```bash
# 首尾帧条件：prepare_fl2va_data.py --frame_mode first_last
FRAME_INDICES='[0,-1]' \
MODEL_PATH=/path/to/MiniMax-H3 DATA_DIR=/path/to/h3_fl2va_first_last \
bash examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh
```

验证时，除了检查 reward，还要检查首帧身份是否保持、prompt 场景是否成立，以及 107 帧内的时间一致性。

## 7. 实验结果

以下曲线来自 T2VA 的在线训练 run，重点用于说明“优化信号和系统路径是否正常”，而不是证明“模型已经在所有任务上超过 base model”。FL2VA 则通过首帧条件下的视频对比检查条件帧链路。

### 7.1 训练 reward 曲线

训练端的平均 reward 从约 0.27 稳步上升到 0.4 以上，说明在当前 prompt 分布下，CLAP + ImageBind 的联合信号能够形成可学习的组内偏好。该指标只能说明 reward model 偏好的方向得到了优化，不能单独解释为画面美学或长程一致性有所提升。

![训练 reward 曲线](/assets/figures/2026-09-18-minimax-h3-rl/train-reward.png)

actor 的动态需要与 reward 结合分析。尤其要关注梯度范数（grad norm）、reward probability 和 reference KL（ref KL）：如果 reward 上升，但梯度范数持续异常、reward probability 饱和，或者 ref KL 突然失效，曲线仍可能收敛到错误目标。

![actor 训练动态](/assets/figures/2026-09-18-minimax-h3-rl/actor-training-dynamics.png)

### 7.2 验证 reward 分解

验证集分别记录 CLAP、ImageBind 和 weighted reward。**只有两个子 reward 的变化方向与固定样本视频一致时，combined reward 才具有解释价值。** 例如，如果 combined reward 的提升只来自 CLAP，可能说明声音更贴合文本，但不能证明音画关系或视觉质量也一同提升。

![评估 reward 曲线](/assets/figures/2026-09-18-minimax-h3-rl/eval-reward.png)

### 7.3 训练耗时分析：rollout 和 reward 的吞吐瓶颈

端到端耗时拆分为 rollout、reward、actor update 和 checkpoint。

![各阶段耗时](/assets/figures/2026-09-18-minimax-h3-rl/time-consumption.png)

从耗时分解来看，rollout 和 reward 合计占据端到端耗时的大头，actor update 和 checkpoint 的占比相对较小。因此，端到端优化需要降低单条 rollout 的耗时、提高 rollout 吞吐，并使 reward 吞吐与之匹配。

### 7.4 视频对比：Base Model 与 DiffusionNFT（T2VA）

reward 曲线只能反映数值层面的整体趋势，最终仍需回到生成结果本身。我们从测试集中选取了四条 prompt，在**相同 seed、相同推理配置**下，分别使用 MiniMax-H3 base model 和 DiffusionNFT 微调后的模型生成音视频输出。观察维度集中在以下三点：

- **motion consistency（运动一致性）**：物体轨迹是否连贯，是否存在抖动或断裂；
- **audio-visual alignment（音画对齐）**：音效出现的时机与画面动作是否同步；
- **视觉细节稳定性**：材质、光照和边缘在整段时长内是否稳定。

左列是 base model，右列是 DiffusionNFT 微调后的模型；prompt 保留原始英文，没有改写。

| ID | Prompt | MiniMax H3 (base) | MiniMax H3 + DiffusionNFT |
|---:|---|---|---|
| 1 | stickman monigote shooting a energy sphere from his hands | [01-stickman-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/01-stickman-base.mp4) | [01-stickman-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/01-stickman-DiffusionNFT.mp4) |
| 2 | a husky dog with sunglasses riding on santas sled | [02-husky-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/02-husky-base.mp4) | [02-husky-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/02-husky-DiffusionNFT.mp4) |
| 3 | minimalist polygonal human skull in green flames with strong movement, uhd | [03-skull-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/03-skull-base.mp4) | [03-skull-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/03-skull-DiffusionNFT.mp4) |
| 4 | 17th century sailing ship making a path through the waves during a storm | [04-ship-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/04-ship-base.mp4) | [04-ship-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/04-ship-DiffusionNFT.mp4) |

### 7.5 视频对比：FL2VA 首帧条件下 Base 与 DiffusionNFT

T2VA 只约束文本和音视频的语义关系，FL2VA 还要求模型服从给定的首帧图像。因此，对比的重点不只是“画面是否美观”，而是**在首帧固定的前提下，模型能否更加连贯地补全后续运动和音轨**。我们从测试集中选取两条 prompt，使用同一张 FLUX.1-dev 首帧作为条件，在**相同 seed、相同推理配置**下，分别使用 base model 和 FL2VA DiffusionNFT 微调后的模型续写音视频。

- **首帧一致性**：生成结果是否忠实衔接给定的条件图，而不是重新绘制新的画面；
- **motion consistency（运动一致性）**：从首帧展开的动作是否连贯，是否存在抖动或漂移；
- **audio-visual alignment（音画对齐）**：滴落、水声等音效的出现时机与画面动作是否同步。

第二列是条件首帧，中间两列分别是 base model 和 FL2VA DiffusionNFT 微调后的输出；prompt 保留原始英文并附中文，没有改写。

| ID | 条件首帧 | Prompt | MiniMax H3 (base) | MiniMax H3 + FL2VA DiffusionNFT |
|---:|---|---|---|---|
| 23 | ![红瓶条件首帧](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-23-bottle-condition.jpg) | Shows a close-up of a woman holding a red bottle with a blue substance dripping from it.<br />一名女子手持红瓶子的特写，蓝色液体正从瓶中滴落。 | [fl2va-23-bottle-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-23-bottle-base.mp4) | [fl2va-23-bottle-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-23-bottle-DiffusionNFT.mp4) |
| 56 | ![水池条件首帧](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-56-pool-condition.jpg) | Shows a man wearing a white shirt, brown apron, and a white hat standing in a pool filled with water.<br />一名穿白衬衫、棕色围裙、戴白帽的男子站在装满水的池子里。 | [fl2va-56-pool-base.mp4](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-56-pool-base.mp4) | [fl2va-56-pool-DiffusionNFT.mp4](/assets/figures/2026-09-18-minimax-h3-rl/fl2va-56-pool-DiffusionNFT.mp4) |

需要说明的是，这里展示的是 FL2VA 集成验证阶段的定性对比，样本量有限。它只能说明条件帧链路和后训练更新确实在生成结果上带来了可见差异，不能作为条件生成质量的完整 benchmark。

## 8. 经验总结

本文围绕两条主线展开：rollout 性能和训推一致性。下面分别总结两条主线上的关键经验。

### 8.1 rollout 性能：rollout 和 reward 的吞吐需要协同

- **rollout 是主要生成瓶颈，reward 是联合音视频场景中的伴生瓶颈**，端到端优化需要匹配两者的吞吐。reward 阶段同时承担音频解码、视频处理和两个 scorer，不能只看 actor 的 GPU 利用率。
- **TP 配置需要在吞吐和 actor-to-rollout 同步显存之间权衡**：T2VA 使用 TP=2；FL2VA 因条件帧额外占用显存，提高到 TP=4，为 LoRA 同步留出余量。

### 8.2 训推一致性：能够运行但实际出错的静默错误

接入 H3 的过程中，最危险的不是导致程序报错中断的错误，**而是“能够运行但实际出错”的错误**——loss 有数值，曲线在变化，但梯度方向反了，或者更新的权重根本没有生效。这些错误几乎都发生在训推一致性链路上。我们按照 rollout 链路顺序逐一排查：

| 阶段 | 表面现象 | 实际问题 | 解决方法 |
|---|---|---|---|
| 训练 adapter 接入 | loss 有数值 | H3 的 timestep / velocity 定义与通用 flow-matching 相反 | 按 H3 约定换算 timestep，并统一 velocity 符号 |
| rollout adapter 接入 | 能够生成音视频 | 训练端和 rollout 端的权重布局不同 | 捕获 clean latents，完成 fused DiT 映射 |
| 同步 LoRA | adapter 注册成功 | vLLM 侧融合后的注意力 / 前馈权重没有真正命中 | 将每一路投影显式映射到融合后的结构，不遗漏任何切片 |
| prompt 接入 | 文本能够传入 | 文本先解码再重新分词，可能导致 prompt 漂移 | 使用 H3 原生文本链路，异常时直接报错中止 |
| audio reward 接入 | 总 reward 有数值 | audio 可能根本没有进入 CLAP / ImageBind | 拆分音视频联合输出，将 audio 一路透传给 reward，并导出带音轨的 MP4 |

表中各行就是我们遇到问题的顺序：每一环都“看似正确，实际错误”，根源都在于训练引擎和推理引擎之间的差异。

总之，对 MiniMax-H3 进行 RL 后训练时，应先逐项验证 rollout 性能和训推一致性链路，再讨论垂直场景中的效果调优。

MiniMax-H3 DiffusionNFT 的训练实践已经全部开源，可以通过文末的参考入口复现。

## 参考入口

- [DiffusionNFT 论文：Online Diffusion Reinforcement with Forward Process](https://arxiv.org/abs/2509.16117)
- [PR #383：MiniMax H3 DiffusionNFT T2VA training](https://github.com/verl-project/verl-omni/pull/383)
- [PR #401：MiniMax H3 FL2VA DiffusionNFT](https://github.com/verl-project/verl-omni/pull/401)
- [MiniMax-H3 T2VA/FL2VA recipe](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/README.md)
- [H3 T2VA launch script](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh)
- [H3 FL2VA launch script](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh)
- [MiniMax-H3 Model](https://huggingface.co/MiniMaxAI/MiniMax-H3)
