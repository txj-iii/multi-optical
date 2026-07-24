# Train目录说明

训练、版本、数据清单和验收标准统一以
[训练学习规则.md](D:/multi-optical/readme/训练学习规则.md) 为准。

## 当前代码入口

| 文件 | 用途 |
| --- | --- |
| `six_band_dataset.py` | 五张相机BMP导入、场景生成、CVAT回写和patch导出 |
| `prepare_background4_training.py` | background4冻结基础集准备与检查 |
| `prepare_background4_v3_training.py` | v3增量集、统一清单与硬约束检查 |
| `run_training.py` | 当前正式训练CLI |
| `vnir_train.py` | 数据集、均衡采样、loss和训练循环 |
| `model.py` | baseline / attention / task_specific模型结构 |
| `five_band_features.py` | 五波段到15通道光谱特征 |
| `predict.py` | 单checkpoint预测和overlay导出 |
| `evaluate_background4.py` | background4评估 |
| `analysis_workbench.py` | UI manifest生成 |
| `workbench_workflow.py` | UI导入、背景确认、预测、标注和状态刷新 |
| `compose_predictions.py` | 论文历史分头结果组合 |
| `export_curves.py` | 五波段曲线导出 |
| `export_dual_pigment_analysis.py` | 颜料子区分析 |

## 当前输入结构

论文历史版本使用15通道：

```text
5原始 + 4差分 + 4比值 + 2归一化差异
```

background4使用19通道：

```text
15通道光谱特征 + 4背景one-hot常量图
```

## 当前训练入口

### background4_v3

```powershell
python train/run_training.py --background4-v3 --epochs 10
```

### background4_v3 agingfix

```powershell
python train/run_training.py --background4-v3-agingfix --epochs 5
```

## 当前预测入口

```text
train/predict.py
```

background4预测必须提供：

- checkpoint
- 样本根目录
- 输出目录
- sample_id
- background_role

详细命令见：

[v3与v3微调版预测命令.md](D:/multi-optical/readme/v3与v3微调版预测命令.md)

## 当前数据目录

```text
camera_eval_workspace/             完整场景和人工masks
five_band_patches/background4_v1/  40场景、1,500 patch冻结基础集
five_band_patches/background4_v3/  23场景、846 patch增量集
experiments/five_band_train/       checkpoint与日志
experiments/five_band_predictions/ 各版本预测
```

`background4_v1` patch 根仍被v2/v3使用，不能删除。

## 当前正式权重

```text
experiments/five_band_train/task_specific/background4_v2/background4_v2.pt
experiments/five_band_train/task_specific/background4_v3/background4_v3.pt
experiments/five_band_train/task_specific/background4_v3_agingfix_v1/background4_v3_agingfix_v1_best.pt
```

## 删除纪律

可以删除逐epoch中间checkpoint、非最佳agingfix预测和可再生成缓存。

不能删除：

- `camera_eval_workspace`
- 人工masks
- background4冻结patch
- 当前最终/最佳checkpoint
- 当前正式预测
- 论文version 1/version 2最终来源

