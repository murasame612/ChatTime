# ChatTime 中文说明

## 项目简介

ChatTime 是一个把时间序列建模为“语言”的多模态时间序列基础模型。它支持三类常见能力：

- 零样本时间序列预测
- 带文本上下文的时间序列预测
- 时间序列问答

原始论文与英文说明见 [README.md](/home/murasame/nas/pythonproject/ChatTime/README.md)。

当前仓库除了原始能力外，还补充了本地 `dam_1h` 数据的微调、评测和多卡启动脚本，便于直接在本地训练。

## 仓库结构

- [model/model.py](/home/murasame/nas/pythonproject/ChatTime/model/model.py)：推理封装，提供 `predict` 和 `analyze`
- [utils/prompt.py](/home/murasame/nas/pythonproject/ChatTime/utils/prompt.py)：Prompt 模板
- [training/pretrain.py](/home/murasame/nas/pythonproject/ChatTime/training/pretrain.py)：持续预训练
- [training/finetune.py](/home/murasame/nas/pythonproject/ChatTime/training/finetune.py)：监督微调主脚本
- [training/build_dam_finetune_dataset.py](/home/murasame/nas/pythonproject/ChatTime/training/build_dam_finetune_dataset.py)：构造 `dam_1h` 微调数据
- [training/evaluate_dam_model.py](/home/murasame/nas/pythonproject/ChatTime/training/evaluate_dam_model.py)：评测微调后的模型

## 环境与模型

你至少需要准备：

- Python 环境
- CUDA 与 PyTorch
- 本地或 Hugging Face 模型权重

当前本地脚本默认使用：

- Chat 模型目录：`./ChatTime-1-7B-Chat`
- dam 数据文件：`./dam_dataset/Dam/DamProcess_1h.csv`

如果依赖未安装，可参考仓库中的：

- [requirements.txt](/home/murasame/nas/pythonproject/ChatTime/requirements.txt)
- [requirements-cu124.txt](/home/murasame/nas/pythonproject/ChatTime/requirements-cu124.txt)
- [requirements-rebuild-working.txt](/home/murasame/nas/pythonproject/ChatTime/requirements-rebuild-working.txt)

## 基本用法

### 1. 零样本预测

```python
import numpy as np
import pandas as pd
from model.model import ChatTime

dataset = "Traffic"
hist_len = 120
pred_len = 24
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
hist_data = np.array(df["Hist"].apply(eval).values.tolist())[:, -hist_len:][0]

model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)
pred = model.predict(hist_data)
print(pred)
```

### 2. 带上下文预测

```python
import numpy as np
import pandas as pd
from model.model import ChatTime

dataset = "PTF"
hist_len = 120
pred_len = 24
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
hist_data = np.array(df["Hist"].apply(eval).values.tolist())[:, -hist_len:][0]
context = df["Text"].values[0]

model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path)
pred = model.predict(hist_data, context)
print(pred)
```

### 3. 时间序列问答

```python
import numpy as np
import pandas as pd
from model.model import ChatTime

dataset = "TSQA"
model_path = "ChengsenWang/ChatTime-1-7B-Chat"

df = pd.read_csv(f"./dataset/{dataset}.csv")
series = np.array(df["Series"].apply(eval).values.tolist())[0]
question = df["Question"].values[0]

model = ChatTime(model_path=model_path)
answer = model.analyze(question, series)
print(answer)
```

## 当前本地 dam 微调流程

这部分是当前仓库里已经整理好的本地工作流，适合直接拿来训练 `dam_1h`。

### 1. 数据构造

脚本：
[training/build_dam_finetune_dataset.py](/home/murasame/nas/pythonproject/ChatTime/training/build_dam_finetune_dataset.py)

作用：

- 从 `DamProcess_1h.csv` 读取数据
- 对每个 `dx*` 列单独构造样本
- 用目标列自己的历史窗口作为输入
- 用目标列未来窗口作为监督输出
- 把上下文拼成文本 prompt

当前默认策略很重要：

- `context_feature_scope=target_only`
- 也就是上下文只保留 `dx*` 目标列的快照
- 不再把所有特征都拼进去

这样做的好处是：

- 更贴近原始 ChatTime 的单目标建模思路
- prompt 明显更短
- 更不容易超过 `max_seq_length`

手动构造数据集示例：

```bash
python training/build_dam_finetune_dataset.py \
  --input_csv dam_dataset/Dam/DamProcess_1h.csv \
  --output_path dataset/dam_1h_dx_sft \
  --hist_len 96 \
  --pred_len 48 \
  --stride 24 \
  --target_prefix dx \
  --context_feature_scope target_only
```

如果你想看旧版“把所有特征都塞进上下文”的模式，也可以显式指定：

```bash
--context_feature_scope all
```

### 2. 单卡微调

完整数据单卡脚本：
[training/finetune_dam_1h.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune_dam_1h.sh)

小样本烟雾测试脚本：
[training/finetune_dam_1h_small.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune_dam_1h_small.sh)

直接运行：

```bash
bash training/finetune_dam_1h_small.sh
```

或者：

```bash
bash training/finetune_dam_1h.sh
```

指定单卡：

```bash
GPU_ID=0 bash training/finetune_dam_1h.sh
```

这两个脚本会自动做三件事：

- 如果需要，先构造数据集
- 调用 [training/finetune.py](/home/murasame/nas/pythonproject/ChatTime/training/finetune.py) 做 LoRA 微调
- 训练结束后调用 [training/evaluate_dam_model.py](/home/murasame/nas/pythonproject/ChatTime/training/evaluate_dam_model.py) 做评测

### 3. 多卡微调

多卡脚本：
[training/finetune_dam_1h_multigpu.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune_dam_1h_multigpu.sh)

这个脚本本质上是一个启动器：

- 负责设置 `CUDA_VISIBLE_DEVICES`
- 负责根据卡数拼 `torchrun`
- 负责在训练前构造数据、训练后评测

真正的训练逻辑仍然在：

- [training/finetune.py](/home/murasame/nas/pythonproject/ChatTime/training/finetune.py)

双卡示例：

```bash
GPU_IDS=0,1 bash training/finetune_dam_1h_multigpu.sh
```

也可以这样传：

```bash
bash training/finetune_dam_1h_multigpu.sh 0,1
```

四卡示例：

```bash
GPU_IDS=0,1,2,3 bash training/finetune_dam_1h_multigpu.sh
```

常见可调参数：

```bash
GPU_IDS=0,1 \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 \
MAX_SEQ_LENGTH=2048 \
NUM_TRAIN_EPOCHS=3 \
bash training/finetune_dam_1h_multigpu.sh
```

## 关于上下文长度

这里容易混淆的一点是：

- 上下文长度不是只由显存决定
- 还受到模型窗口和训练脚本里的 `max_seq_length` 约束

在当前仓库中，[training/finetune.py](/home/murasame/nas/pythonproject/ChatTime/training/finetune.py) 会把 `max_seq_length` 传给 Unsloth 和 `SFTTrainer`。如果样本 token 数超过这个值，就会被截断。

之前把所有特征都作为文本上下文时，单条样本会非常长。现在默认切到 `target_only`，就是为了先用最简单稳定的方式把样本长度控制住。

## 评测

评测脚本：
[training/evaluate_dam_model.py](/home/murasame/nas/pythonproject/ChatTime/training/evaluate_dam_model.py)

手动运行示例：

```bash
python training/evaluate_dam_model.py \
  --model_path ./outputs/dam_1h_dx \
  --dataset_path ./dataset/dam_1h_dx_sft \
  --split validation \
  --output_path ./outputs/dam_1h_dx/eval_validation.json \
  --max_samples 100
```

默认会输出：

- `MSE`
- `MAE`
- `RMSE`

并额外保存逐样本预测记录。

### 数据集切分说明

当前 `dam_1h` 数据集构造脚本已经改成：

- 每个 `dx*` 目标通道先各自按时间顺序生成窗口
- 每个通道内部独立切分 `train / validation / test`
- 再把所有通道对应 split 合并

也就是说，现在不再是把所有通道样本先混在一起再整体按比例切分。

对应实现见：
[training/build_dam_finetune_dataset.py](/home/murasame/nas/pythonproject/ChatTime/training/build_dam_finetune_dataset.py)

生成后的 `metadata.json` 里也会记录：

- `split_strategy: per_target_time_order`
- `per_target_num_samples`

如果你之前已经生成过旧版 `dataset/dam_1h_dx_sft`，建议删除后重新构建一次数据集，再继续训练和评测。

### 预测结果可视化

可视化脚本：
[tools/visualize_eval_windows.py](/home/murasame/nas/pythonproject/ChatTime/tools/visualize_eval_windows.py)

这个脚本直接读取：

- `eval_*.json`
- `eval_*_predictions.json`

并支持三种模式：

- `single`：单窗口检查
- `overlap`：连续多个窗口重叠显示
- `stitch_center`：针对 `pred_len > stride` 的中心区间拼接

建议使用顺序：

1. 先看单窗口，确认 `hist / true / pred` 对齐没问题
2. 再看重叠窗口，检查相邻窗口在重叠区是否稳定
3. 最后再看中心区间拼接图

单窗口示例：

```bash
python tools/visualize_eval_windows.py \
  --eval_path outputs/dam_1h_dx/eval_validation.json \
  --target_col dx_IN1-1-10M \
  --mode single \
  --window_indices 0
```

重叠窗口示例：

```bash
python tools/visualize_eval_windows.py \
  --eval_path outputs/dam_1h_dx/eval_validation.json \
  --target_col dx_IN1-1-10M \
  --mode overlap \
  --window_indices 0,1,2,3,4
```

中心拼接示例：

```bash
python tools/visualize_eval_windows.py \
  --eval_path outputs/dam_1h_dx/eval_validation.json \
  --target_col dx_IN1-1-10M \
  --mode stitch_center \
  --window_indices 0,1,2,3,4 \
  --center_width 24
```

对于 `hist_len=92, pred_len=48, stride=24` 这类半重叠窗口，更推荐 `stitch_center`，不要直接把整段 `48` 步未来硬拼接。

## 原始训练入口

仓库原本还带有两个更通用的脚本：

- [training/pretrain.sh](/home/murasame/nas/pythonproject/ChatTime/training/pretrain.sh)
- [training/finetune.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune.sh)

这两个脚本更接近原项目的持续预训练和指令微调流程，但当前写法仍然偏单卡启动。

## 补充文档

如果你想看更细的本地改动说明，可以继续看：

- [DOCS/DOCS.md](/home/murasame/nas/pythonproject/ChatTime/DOCS/DOCS.md)
- [DOCS/repo_guide.html](/home/murasame/nas/pythonproject/ChatTime/DOCS/repo_guide.html)

## 引用

如果这个项目对你的研究有帮助，可以引用原论文：

```tex
@inproceedings{
  author    = {Chengsen Wang and Qi Qi and Jingyu Wang and Haifeng Sun and Zirui Zhuang and Jinming Wu and Lei Zhang and Jianxin Liao},
  title     = {ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data},
  booktitle = {AAAI Conference on Artificial Intelligence},
  year      = {2025},
}
```
