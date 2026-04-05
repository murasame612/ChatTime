# ChatTime 项目文档

这份文档用于记录当前仓库中的本地改动、使用方法、已知不足和后续计划。

以后只要新增功能，请优先更新这份文档。

## 当前已新增内容

### 1. `.gitignore`

已在根目录新增 `.gitignore`，当前主要忽略以下内容：

- Python 缓存文件
- 虚拟环境目录
- 编辑器目录
- Notebook / 测试缓存
- 训练日志和检查点
- 模型权重产物，如 `.pt`、`.pth`、`.bin`、`.safetensors`

### 2. 本地数据集加载支持

文件： [training/finetune.py](/home/murasame/nas/pythonproject/ChatTime/training/finetune.py)

原始 `finetune.py` 默认更偏向直接读取 Hugging Face 数据集名。
现在已经扩展为支持本地数据。

当前支持：

- Hugging Face 数据集名称
- 单个 `.json` / `.jsonl` / `.csv` / `.parquet` 文件
- 包含 `train.jsonl` / `train.json` / `train.csv` / `train.parquet` 的本地目录
- 通过 `datasets.save_to_disk` 保存的数据集目录

### 3. `dam_1h` 微调数据构造脚本

文件： [training/build_dam_finetune_dataset.py](/home/murasame/nas/pythonproject/ChatTime/training/build_dam_finetune_dataset.py)

作用：

- 读取 `dam_dataset/Dam/DamProcess_1h.csv`
- 默认只使用 `dx*` 目标列作为上下文特征
- 使用所有 `dx*` 列作为预测目标
- 转换成 ChatTime 当前可训练的 SFT 样本格式

默认参数：

- `hist_len=120`
- `pred_len=24`
- `stride=24`
- `context_feature_scope=target_only`

输出内容：

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `metadata.json`

### 4. `dam_1h` 一键微调脚本

文件： [training/finetune_dam_1h.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune_dam_1h.sh)

作用：

- 如果数据集不存在，先自动构造 `dam_1h` 微调数据
- 再调用 `finetune.py` 对本地模型做 4-bit LoRA 微调

默认假设：

- 数据路径：`dam_dataset/Dam/DamProcess_1h.csv`
- 模型路径：`./ChatTime-1-7B-Chat`
- 微调数据输出路径：`./dataset/dam_1h_dx_sft`
- 微调模型输出路径：`./outputs/dam_1h_dx`

另有多卡版本脚本：

文件： [training/finetune_dam_1h_multigpu.sh](/home/murasame/nas/pythonproject/ChatTime/training/finetune_dam_1h_multigpu.sh)

作用：

- 使用 `torchrun` 启动多卡 LoRA 微调
- 默认使用 `target_only` 上下文构造数据
- 训练完成后自动执行评测

### 5. `dam_1h` 训练后自动评测

文件： [training/evaluate_dam_model.py](/home/murasame/nas/pythonproject/ChatTime/training/evaluate_dam_model.py)

作用：

- 读取 `validation` 或 `test` 数据
- 调用训练后的模型逐条预测
- 自动计算并保存：
  - `MAE`
  - `MSE`
  - `RMSE`

默认行为：

- `finetune_dam_1h.sh` 训练结束后自动执行评测
- 默认评测 `validation`
- 默认最多评测 `100` 条样本，避免评测过久

默认输出文件：

- `outputs/dam_1h_dx/eval_validation.json`
- `outputs/dam_1h_dx/eval_validation_predictions.json`

## 使用方法

### 1. 生成 `dam_1h` 微调数据

在项目根目录执行：

```bash
python training/build_dam_finetune_dataset.py \
  --input_csv dam_dataset/Dam/DamProcess_1h.csv \
  --output_path dataset/dam_1h_dx_sft \
  --hist_len 120 \
  --pred_len 24 \
  --stride 24 \
  --target_prefix dx \
  --context_feature_scope target_only
```

可选调试参数：

- `--limit_targets`
- `--limit_windows_per_target`

这两个参数适合先做小样本测试，确认数据格式没问题。

### 2. 开始微调

在项目根目录执行：

```bash
bash training/finetune_dam_1h.sh
```

如果要双卡或多卡启动，可以执行：

```bash
GPU_IDS=0,1 bash training/finetune_dam_1h_multigpu.sh
```

也可以直接把卡号作为第一个参数传入：

```bash
bash training/finetune_dam_1h_multigpu.sh 0,1
```

如果你之前已经生成过旧版 `dataset/dam_1h_dx_sft`，建议先删除再重建。
因为现在评测依赖数据集中新增的字段：

- `context`
- `hist_data`
- `future_data`

建议这样执行：

```bash
rm -rf dataset/dam_1h_dx_sft
bash training/finetune_dam_1h.sh
```

如需覆盖默认路径和参数，可以通过环境变量传入：

```bash
MODEL_PATH=/path/to/ChatTime-1-7B-Chat \
DATASET_PATH=./dataset/dam_1h_dx_sft \
OUTPUT_PATH=./outputs/dam_1h_dx \
PER_DEVICE_TRAIN_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=16 \
bash training/finetune_dam_1h.sh
```

如果想控制评测行为，也可以额外传：

```bash
EVAL_SPLIT=validation \
EVAL_MAX_SAMPLES=200 \
EVAL_OUTPUT_PATH=./outputs/dam_1h_dx/eval_validation.json \
bash training/finetune_dam_1h.sh
```

### 3. 做小样本烟雾测试

```bash
python training/build_dam_finetune_dataset.py \
  --input_csv dam_dataset/Dam/DamProcess_1h.csv \
  --output_path dataset/dam_1h_dx_sft_smoke \
  --hist_len 120 \
  --pred_len 24 \
  --stride 24 \
  --target_prefix dx \
  --limit_targets 2 \
  --limit_windows_per_target 3
```

### 4. 单独执行评测

如果模型已经训练完成，也可以单独跑评测：

```bash
python training/evaluate_dam_model.py \
  --model_path ./outputs/dam_1h_dx \
  --dataset_path ./dataset/dam_1h_dx_sft \
  --split validation \
  --output_path ./outputs/dam_1h_dx/eval_validation.json \
  --max_samples 100
```

## 当前数据构造策略

当前 `dam_1h` 的样本构造遵循的是“兼容现有 ChatTime 架构”的思路，而不是重写模型结构。

对于每一个目标列：

1. 取该目标列自己的历史窗口，作为主输入序列。
2. 取该目标列自己的未来窗口，作为监督输出。
3. 默认只取其他 `dx*` 目标列在“历史窗口最后一个时刻”的数值，拼成文本上下文。
4. 仅对目标序列本身做 ChatTime 原有的离散化和序列化。

这样做的好处是：

- 不需要重写 ChatTime 核心模型
- 可以直接复用当前 prompt 和训练流程
- 能满足“所有通道都参与输入”的需求

但这不是严格意义上的真正多通道时序建模，见下方不足说明。

## 已知不足

### 1. 这不是严格的多通道时序模型

当前 ChatTime 原始实现本质上更偏向“单序列输入 -> 单序列输出”。

现在的 `dam_1h` 方案是：

- 一次只预测一个 `dx*` 目标序列
- 其余通道以文本上下文的形式注入 prompt

这意味着当前还没有实现：

- 多通道联合序列编码
- 多目标同时解码
- 显式的通道 embedding
- 专门的多变量时间序列编码器

### 2. 非目标特征只用了最后时刻快照

现在所有非目标通道，都是取历史窗口最后一个时刻的值，写入上下文文本。

优点：

- 所有通道都参与了输入
- prompt 长度还比较可控

缺点：

- 模型没有看到这些辅助通道的完整历史轨迹
- 只能利用“最后一时刻的状态信息”

所以这是一种工程上可落地、但表达能力有限的折中方案。

### 3. 当前默认微调基座是 `ChatTime-1-7B-Chat`

现在脚本默认用的是：

- `./ChatTime-1-7B-Chat`

这对本地快速验证方便，但从训练逻辑上讲，继续在 chat checkpoint 上做领域微调，不一定是最干净的选择。

更合理的做法通常是：

- 用 base / pretrain checkpoint 做领域适配
- 再决定是否继续 instruction tuning

### 4. 已有基础评测，但还不完整

目前已经补上了基础评测脚本，可以自动输出：

- `MAE`
- `MSE`
- `RMSE`

但仍然缺少更完整的实验体系，包括：

- 基线方法对比
- 更系统的 test 集全量评测
- 训练过程中的在线验证
- 更完整的可视化报告

### 5. 特征越多，prompt 越长

因为当前方案把所有非目标特征都写进文本上下文，所以特征数增加时，prompt 会继续变长。

在当前 `dam_1h` 规模下仍然可控，但如果后面特征更多、窗口更复杂，这会成为瓶颈。

## 建议下一步补充内容

建议下一步优先补：

1. `dam_1h` 微调后推理脚本
2. `dam_1h` 评估脚本，至少包含 MSE / MAE
3. 基于时间范围而不是比例的 train / val / test 切分
4. 非目标通道的更强上下文表示，例如最近若干步摘要
5. 如果后续追求真正多变量效果，重构输入格式，引入显式通道编码

## 变更记录

### 2026-03-28

- 新增根目录 `.gitignore`
- 扩展 `training/finetune.py`，支持本地数据集加载
- 新增 `training/build_dam_finetune_dataset.py`
- 新增 `training/finetune_dam_1h.sh`
- 已完成 `dam_1h` 小样本数据构造验证

### 2026-03-29

- 重写 `requirements.txt`，改为项目直接依赖列表
- 重写 `requirements.md`，补充推荐安装流程和 CUDA / notebook 注意事项
- 明确补上 `trl`、`unsloth`、`matplotlib`、`peft`、`sentencepiece`、`safetensors` 等关键依赖
- 新增 `training/evaluate_dam_model.py`
- `dam_1h` 微调流程增加训练后自动评测
- 评测结果支持落盘保存 `MAE / MSE / RMSE`
