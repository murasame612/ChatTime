你现在要对当前的时间序列预测代码做一次**最小侵入式改造**。不要重构整个项目，也不要改训练主流程的接口风格。你的任务是在**现有模型输入端前面**插入一个轻量模块：**VSN/Gate + MLP Compressor**，用于对每个时间步的多变量输入进行**可学习特征选择**和**低维压缩**，然后再送入后续预测模型。

### 目标

当前输入张量形状默认为：
[
x \in \mathbb{R}^{B \times T \times D}
]
其中：

* (B): batch size
* (T): 序列长度
* (D): 输入特征维度

现在增加一个前端模块，处理流程为：

1. **Feature Gate（软特征选择）**
   对每个时间步的 (D) 维特征生成一个 gate：
   [
   g_t = \sigma(\text{MLP}(x_t)) \in (0,1)^D
   ]
   然后进行逐元素加权：
   [
   \hat{x}_t = g_t \odot x_t
   ]

2. **Feature Compressor（MLP 压缩）**
   将加权后的每个时间步特征压缩到低维：
   [
   z_t = \text{MLP}(\hat{x}_t) \in \mathbb{R}^{d}
   ]
   其中 (d) 是压缩后的维度，比如 16 或 32。

最终输出张量变为：
[
z \in \mathbb{R}^{B \times T \times d}
]

然后把这个 (z) 送给后续原有模型主干。也就是说，原模型如果以前吃的是 `B,T,D`，现在改为吃 `B,T,d`。

---

### 实现要求

#### 1. 新增模块

请新增两个 PyTorch 模块，或者合并为一个前端模块：

* `FeatureGate`
* `FeatureCompressor`
* 或统一封装成 `GatedFeatureCompressor`

推荐结构：

```python
class FeatureGate(nn.Module):
    # input: [B, T, D]
    # output:
    #   gated_x: [B, T, D]
    #   gate:    [B, T, D]
```

Gate 使用两层 MLP，例如：

* Linear(D, hidden_dim)
* GELU / ReLU
* Linear(hidden_dim, D)
* Sigmoid

Compressor 使用两层或三层 MLP，例如：

* Linear(D, hidden_dim)
* GELU
* Dropout(optional)
* Linear(hidden_dim, compressed_dim)

---

#### 2. 前向传播逻辑

在前端模块中完成以下逻辑：

```python
gate = self.feature_gate(x)          # [B, T, D]
gated_x = x * gate                   # [B, T, D]
z = self.feature_compressor(gated_x) # [B, T, d]
```

返回：

* `z`
* `gate`（方便可视化和做正则）

---

#### 3. 与现有模型集成

请在**现有主模型的输入端**接入该模块，而不是重写整个主模型。

目标形式类似：

```python
class MyForecastModel(nn.Module):
    def __init__(...):
        ...
        self.input_adapter = GatedFeatureCompressor(
            input_dim=D,
            gate_hidden_dim=64,
            compressor_hidden_dim=64,
            compressed_dim=16,
            dropout=0.1,
        )
        self.backbone = ExistingBackbone(...)
```

前向传播：

```python
def forward(self, x, ...):
    z, gate = self.input_adapter(x)
    out = self.backbone(z, ...)
    return out, gate
```

如果现有代码的 `forward()` 原本只返回预测值，请尽量保持兼容：

* 可以增加一个可选参数 `return_gate=False`
* 默认返回原预测结果
* 在需要分析时返回 `(pred, gate)`

---

#### 4. 损失函数改造

请支持一个简单的 gate 稀疏正则项：

[
L = L_{pred} + \lambda \cdot \text{mean}(|g|)
]

即：

* `pred_loss` 保持原有损失（MSE/MAE 等）
* 新增 `gate_l1_loss = gate.abs().mean()`
* 总损失：

```python
loss = pred_loss + gate_l1_weight * gate_l1_loss
```

要求：

* `gate_l1_weight` 可配置，默认例如 `1e-4`
* 如果训练脚本中不方便大改，至少把 `gate_l1_loss` 暴露出来，方便后续接入

---

#### 5. 配置项

请为该模块新增配置参数，尽量接入现有 config / argparse / yaml / toml 体系。至少包含：

* `use_gated_feature_compressor: bool = True`
* `input_dim`
* `gate_hidden_dim: int = 64`
* `compressor_hidden_dim: int = 64`
* `compressed_dim: int = 16`
* `gate_dropout: float = 0.1`
* `gate_l1_weight: float = 1e-4`

如果当前项目已经有统一配置风格，请遵循现有风格，不要另起炉灶。

---

#### 6. 日志与可解释性

请增加一些最基础的日志或调试输出，方便后续汇报：

* 打印输入维度和压缩后维度
* 可选统计 gate 的均值、最小值、最大值
* 保证后续可以很方便拿到 `gate.mean(dim=(0,1))` 作为全局特征重要性参考

不要做复杂可视化，但要给出容易扩展的接口。

---

#### 7. 兼容性要求

* 尽量少改现有训练流程
* 不要破坏单通道场景
* 当 `use_gated_feature_compressor=False` 时，应退化为原始模型逻辑
* 保持 batch-first，即 `[B, T, D]`
* 保持现有 device / dtype 逻辑兼容
* 代码风格与项目现有风格一致
* 给关键位置加简洁注释

---

### 你需要完成的内容

请直接输出以下内容：

1. **需要修改的文件列表**
2. **每个文件的具体改动说明**
3. **完整可运行代码**
4. **如果某些地方依赖项目上下文，请做最合理的最小假设**
5. **不要只给伪代码，要给能直接粘贴的 PyTorch 实现**
6. **如有必要，可顺手补一个简单单元测试或 shape check**

---

### 设计原则

这次改动的目标不是做最先进方法，而是做一个**结构清晰、容易汇报、可解释、可快速跑通的 baseline**。因此请遵守以下原则：

* 优先最小改动
* 优先稳定训练
* 优先接口兼容
* 不要引入复杂注意力模块
* 不要引入 Gumbel、VQ、MoE 等复杂机制
* 不要重写整个 backbone
* 不要做与任务无关的大规模重构

---

### 命名建议

模块名建议使用：

* `GatedFeatureCompressor`
* `FeatureGate`
* `FeatureCompressor`

baseline 名称可在注释中写为：

* `VSN/Gate + MLP Compression baseline`

---

如果你发现现有 backbone 的输入维度写死了，请顺手把它改成可配置的 `input_dim`，使其能够接受压缩后的 `compressed_dim` 输入。

