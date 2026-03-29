# 环境说明

## 硬件环境

- CPU: `AMD Ryzen 9 7950X 16-Core Processor`
- GPU: `NVIDIA GeForce RTX 4090 24GB`

## 系统环境

- Kernel: `Linux-5.15.0-102-generic-x86_64-with-glibc2.35`
- Distribution: `Ubuntu 22.04.3 LTS`
- Python: `3.12.11`

## 当前项目真正需要的依赖

项目代码实际直接用到的核心依赖已经整理到 [requirements.txt](/home/murasame/nas/pythonproject/ChatTime/requirements.txt)：

- `accelerate`
- `datasets`
- `huggingface-hub`
- `matplotlib`
- `numpy`
- `pandas`
- `peft`
- `safetensors`
- `scikit-learn`
- `sentencepiece`
- `transformers`
- `trl`
- `unsloth`

之前的依赖文件更像整机环境导出，不适合直接拿来复现项目环境，而且还漏掉了：

- `trl`
- `unsloth`
- `matplotlib`
- `peft`
- `sentencepiece`
- `safetensors`

这几个依赖里：

- `trl` 和 `unsloth` 是训练脚本必需的
- `matplotlib` 是 README / notebook demo 需要的
- `sentencepiece` 是 Llama tokenizer 常见依赖
- `peft` 和 `safetensors` 是 LoRA / 模型加载常见依赖

## 推荐安装方式

建议按下面顺序装环境。

### 1. 创建虚拟环境

```bash
conda create -n chattime python=3.12 -y
conda activate chattime
```

### 2. 按 CUDA 版本安装 PyTorch

这个项目不要把 `torch` 固定死在 `requirements.txt` 里，因为不同机器的 CUDA 版本不一样。

如果你当前机器是 CUDA 11.8，可以用：

```bash
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

如果不是 CUDA 11.8，请改成你机器对应的官方安装命令。

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 如果要跑 notebook

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name chattime --display-name "Python (chattime)"
```

## 为什么不再直接维护整机导出列表

原因很简单：

- 整机导出列表太长，噪音太多
- 很多包和项目本身无关
- 不同机器上几乎一定会漂移
- 一旦缺少关键训练依赖，反而更难排查

所以现在的策略是：

- `requirements.txt` 只维护项目直接依赖
- `requirements.md` 负责说明安装顺序和 CUDA / notebook 等补充步骤

## 已知注意事项

### 1. `unsloth` 对环境比较敏感

如果你在安装或导入 `unsloth` 时遇到问题，优先检查：

- Python 版本
- PyTorch 版本
- CUDA 版本
- `transformers` / `trl` / `peft` 是否兼容

### 2. `torch` 不能盲装

如果直接 `pip install torch`，很可能装成 CPU 版，或者和本机 CUDA 不匹配。
训练前请先确认：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. notebook 和终端环境可能不是同一个

如果终端能导包、notebook 不行，通常不是依赖没装，而是 notebook kernel 用错了环境。

## 2026-03-29 更新

- 重写 `requirements.txt`，改为项目直接依赖
- 重写 `requirements.md`，补充推荐安装流程
- 补上训练脚本缺失的关键依赖说明
