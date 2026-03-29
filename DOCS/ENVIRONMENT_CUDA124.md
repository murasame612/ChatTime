# CUDA 12.4 环境建议

这份说明用于在 `CUDA 12.4 + NVIDIA Driver 550` 环境下，重建一套比当前更稳的 ChatTime 训练环境。

## 结论先说

如果你当前机器是：

- CUDA Runtime / Toolkit: `12.4`
- NVIDIA Driver: `550.x`

更稳的建议是：

- Python：`3.11`
- PyTorch：`2.5.1 + cu124`
- torchvision：`0.20.1`
- torchaudio：`2.5.1`
- Unsloth：按官方 `cu124-torch250` 方式安装
- 先接受 `xformers` 跑通训练，不把 `flash-attn` 作为第一优先级

我这里的判断基于：

- PyTorch 官方 CUDA 12.4 安装说明
- Unsloth 官方针对 `torch 2.5 + cu124` 的安装说明

## 为什么推荐这套

你现在的主要问题不是单个包缺失，而是下面几件事叠在一起：

- `torch / CUDA / unsloth / trl / transformers` 之间版本联动很强
- Python 3.12 下某些包更容易踩兼容问题
- `flash-attn` 在不同组合下安装成本高，容易干扰主流程

所以现在更好的策略不是“继续往旧环境里打补丁”，而是直接切到一套更保守、更贴近官方说明的组合。

## 官方依据

### 1. PyTorch 官方支持 CUDA 12.4

PyTorch 官方 “previous versions” 页面给出了 `2.5.1` 的 `cu124` 安装方式。  
参考：

- https://pytorch.org/get-started/previous-versions
- https://docs.pytorch.org/get-started/previous-versions/

对应官方 pip 安装命令：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 2. Unsloth 官方支持 `cu124-torch250`

Unsloth 官方安装文档给出了 `torch 2.5 + CUDA 12.4` 的高级 pip 安装方式。  
参考：

- https://docs.unsloth.ai/get-started/install-and-update/pip-install
- https://github.com/unslothai/unsloth

对应安装形式：

```bash
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

## 推荐重建步骤

建议新建一个全新的环境，不要在当前已经打架的环境上继续修。

### 1. 创建新环境

```bash
conda create -n chattime-cu124 python=3.11 -y
conda activate chattime-cu124
```

### 2. 安装 PyTorch CUDA 12.4

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装项目基础依赖

```bash
pip install -r requirements-cu124.txt
```

### 4. 安装 Unsloth

```bash
pip install --upgrade pip
pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

如果你只想先快速跑通，也可以先试：

```bash
pip install unsloth
```

但对 `cu124 + torch 2.5`，更推荐上面的官方定制安装方式。

### 5. 可选：安装 notebook

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name chattime-cu124 --display-name "Python (chattime-cu124)"
```

## `flash-attn` 怎么看

目前不建议把 `flash-attn` 作为第一优先级。

原因：

- 你现在的主要目标是稳定训练和稳定评测
- `flash-attn` 安装对 `torch / CUDA / 编译环境` 很敏感
- Unsloth 已经可以退回 `xformers`
- 对你当前这类实验，先跑通比先追极限速度更重要

建议顺序：

1. 先在 `xformers` 下把训练、评测、导出全跑通
2. 再单独尝试 `flash-attn`

## 推荐检查命令

环境装完后，先检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import transformers, trl, accelerate; print(transformers.__version__, trl.__version__, accelerate.__version__)"
python -c "import unsloth; print('unsloth ok')"
```

## 这套环境的取舍

### 优点

- 更贴近 PyTorch 官方 CUDA 12.4 支持范围
- 更贴近 Unsloth 官方 `cu124-torch250` 说明
- Python 3.11 通常比 3.12 更稳
- 适合把当前工程先稳定下来

### 不足

- 不保证 `flash-attn` 一定一步装好
- `trl` 新版本接口变化较快，所以项目脚本仍然保留了兼容逻辑
- 如果未来升级更高版本 `torch / trl / transformers`，仍可能需要微调脚本

## 适合你的实际建议

如果你现在最优先的是“尽快稳定出结果”，建议直接这样做：

1. 新建 `chattime-cu124` 环境
2. 按本文步骤安装
3. 用 `small` 数据先跑通训练和评测
4. 确认 `MAE / MSE / RMSE` 能稳定产出
5. 再回头考虑 `flash-attn`
