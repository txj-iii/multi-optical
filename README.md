# Multi-Optical：五波段壁画表面病害与颜料分析

本项目使用 `450 / 550 / 600 / 650 / 700 nm` 五波段图像，完成：

- `paint`：颜料或绘制主体分割
- `pollution`：外来污染与附着物分割
- `aging`：老化、褪变等病害分割
- `pigment`：paint 内朱砂、代赭、石青、石绿四类像素级颜料识别

## 当前模型

项目保留两套不能混称的版本体系。

| 体系 | 输入 | 用途 |
| --- | ---: | --- |
| 论文 version 1 / version 2 | 15通道 | 论文基线、soft competition与历史对照 |
| background4 v2 / v3 / v3 agingfix | 19通道 | 当前四背景条件推理与像素级pigment |

当前 background4 链：

```text
background4_v2
  -> background4_v3
  -> background4_v3_agingfix_v1_best
```

- UI 的 background4 下拉框默认使用 v2。
- v3 是吸收 `SAMPLE_097–119` 后的正式候选。
- v3 agingfix 是修复 aging 与背景误报的最佳第4轮候选。

完整版本差异、训练数据、指标和文件保留规则见
[训练学习规则.md](D:/multi-optical/readme/训练学习规则.md)。

## 当前UI链路

```text
5张相机BMP
-> five_band.npy + preview.png
-> 选择并保存背景板
-> 选择background4版本
-> 预测与人工复核
```

UI 同时保留论文 version 2 的 softcomp review 起标链，用于生成可编辑标注草稿。

工作台地址：

```text
http://127.0.0.1:8768/ui/analysis_workbench/index.html
```

当前UI目录：

```text
D:\multi-optical\ui\analysis_workbench
```

`D:\multi-optical\1` 是旧备份，不是当前UI入口。

## 项目目录

```text
train/
├── camera_eval_workspace/        # 完整场景、preview、人工masks
├── five_band_patches/            # 训练patch
├── experiments/
│   ├── five_band_train/          # checkpoint和训练日志
│   └── five_band_predictions/    # 各版本预测
├── six_band_dataset.py           # BMP导入与场景生成
├── run_training.py               # 训练入口
├── predict.py                    # 预测入口
├── model.py                      # 模型结构
├── vnir_train.py                 # 数据、采样与loss
├── analysis_workbench.py         # UI manifest生成
└── workbench_workflow.py         # UI工作流

ui/
└── analysis_workbench/           # 当前本地UI

readme/
├── 训练学习规则.md                # 唯一训练与版本口径
├── 样本记录规范.md                # 样本、分组与标注语义
└── v3与v3微调版预测命令.md        # PyCharm终端命令
```

## 样本输入

每个待预测样本至少需要：

```text
SAMPLE_xxx/
├── five_band.npy
└── preview.png
```

`five_band.npy` 必须为 `H × W × 5`，波段顺序固定为：

```text
450 / 550 / 600 / 650 / 700 nm
```

background4 推理还必须指定：

```text
--background-role 代赭 / 石青 / 石绿 / 朱砂
```

## 预测输出

每个样本通常包含：

```text
paint_pred.png
pollution_pred.png
aging_pred.png
pigment_pred.png
paint_overlay.png
pollution_overlay.png
aging_overlay.png
combined_overlay.png
aging_probability.png
pigment_summary.json
```

## 常用入口

### v3或v3微调版预测

命令统一见：

[v3与v3微调版预测命令.md](D:/multi-optical/readme/v3与v3微调版预测命令.md)

### 启动UI

先激活安装了 PyTorch 的 Conda 环境：

```powershell
conda activate pipe2
$env:WORKBENCH_PYTHON="$env:CONDA_PREFIX\python.exe"
node D:\multi-optical\ui\analysis_workbench\server.mjs D:\multi-optical 8768
```

相机目录不在默认位置时设置：

```powershell
$env:CAMERA_IMAGE_ROOT="你的相机BMP目录"
```

## 当前重要文件

### Checkpoint

```text
train/experiments/five_band_train/task_specific/background4_v2/background4_v2.pt
train/experiments/five_band_train/task_specific/background4_v3/background4_v3.pt
train/experiments/five_band_train/task_specific/background4_v3_agingfix_v1/background4_v3_agingfix_v1_best.pt
```

### 预测目录

```text
train/experiments/five_band_predictions/task_specific/background4_v2
train/experiments/five_band_predictions/task_specific/background4_v3
train/experiments/five_band_predictions/task_specific/background4_v3_agingfix_v1_best
```

### 训练patch

```text
train/five_band_patches/background4_v1
train/five_band_patches/background4_v3
```

`background4_v1` patch 根仍是v2/v3冻结基础集，不能因名称较旧而删除。

## 文档入口

- [训练学习规则.md](D:/multi-optical/readme/训练学习规则.md)：版本、训练、预测、验收和保留规则
- [样本记录规范.md](D:/multi-optical/readme/样本记录规范.md)：逐样本状态、split和标注语义
- [当前主线说明.md](D:/multi-optical/readme/当前主线说明.md)：当前链路摘要
- [ui/README.md](D:/multi-optical/ui/README.md)：UI操作与状态文件
- [train/README.md](D:/multi-optical/train/README.md)：训练代码入口
- [迷你主机CPU部署清单.md](D:/multi-optical/readme/迷你主机CPU部署清单.md)：CPU迁移与部署
