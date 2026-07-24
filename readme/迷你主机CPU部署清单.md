# 迷你主机 CPU 推理与 UI 迁移清单

本文只描述另一台 **Anaconda + CPU** 电脑上的预测、相机样本导入和 UI 查看链路，不包含训练环境。当前应迁移的是 19 通道 `background4` 链路，不要再接旧的 15 通道论文模型或 `agingonly` 临时链路。

## 1. 必须迁移的模型

至少迁移准备实际使用的权重：

| 版本 | 文件 |
| --- | --- |
| `background4_v3` | `train/experiments/five_band_train/task_specific/background4_v3/background4_v3.pt` |
| `background4_v3_agingfix_v1` | `train/experiments/five_band_train/task_specific/background4_v3_agingfix_v1/background4_v3_agingfix_v1_best.pt` |

如需在 UI 中切换或回退，再补充：

| 版本 | 文件 |
| --- | --- |
| `background4_v2` | `train/experiments/five_band_train/task_specific/background4_v2/background4_v2.pt` |

权重必须与当前代码一起迁移。仅复制 `.pt`、但继续使用旧电脑上的 15 通道代码，会出现加载失败或预测语义错误。

## 2. 必须迁移的推理代码

保留仓库内相对路径，复制以下文件：

```text
train/predict.py
train/model.py
train/five_band_features.py
train/vnir_train.py
train/six_band_dataset.py
```

如果另一台电脑只接收整理好的 `five_band.npy + preview.png`，以上文件足够支撑命令行预测。

如果接收相机 BMP 原图，还必须保留 `train/six_band_dataset.py` 的相机导入链路，并安装 `spectral`。导入后应得到 UI/推理可识别的五波段场景目录，而不是直接把 BMP 交给 `predict.py`。

## 3. 使用 UI 时必须迁移的文件

复制：

```text
ui/analysis_workbench/
train/workbench_workflow.py
train/analysis_workbench.py
train/compose_predictions.py
train/pigment_subtype.py
train/pigment_subtype_common.py
train/pigment_subtype_train.py
```

以当前 `ui/analysis_workbench` 为准；根目录下的 `1` 是旧备份，不应覆盖当前 UI。

如果目标电脑已有 UI，不建议零散替换一个前端文件。应整体替换 `ui/analysis_workbench/`，再同步上面的 Python 链路，以免前端版本选项、API 和后端参数不一致。

## 4. 样本是否必须迁移

只做新样本预测时，训练样本和验证集不是运行必需项。建议额外携带一小组已确认的复杂样本，用于迁移验收：

```text
five_band.npy
preview.png
人工标签（如有）
已确认预测结果（作为对照）
样本元数据
```

不要用 `SAMPLE_094–096` 作为定量验收集：它们没有可用的正式真值，而且历史上存在背景归属冲突。迁移验收应选择背景明确、人工标注完整、当前版本识别效果已确认的复杂样本。

若要把现有 UI 中的样本继续带过去，需要整体复制相应场景目录，并核对清单/状态文件中的绝对路径。不要直接覆盖目标电脑已有的 `workflow_state.json` 或样本清单；应先备份，再按 sample_id 合并。

## 5. Anaconda CPU 环境

推荐创建独立环境：

```powershell
conda create -n multi-optical-cpu python=3.11 -y
conda activate multi-optical-cpu
```

当前已验证环境的主要版本为：

```text
Python 3.11.13
torch 2.0.1
torchvision 0.15.2
numpy 1.24.4
Pillow 11.3.0
scipy 1.10.0
spectral 0.24
segmentation-models-pytorch 0.5.0
```

CPU 电脑安装 PyTorch 时应选择 CPU 构建。其余依赖按当前代码实际导入补齐。版本不必逐字一致，但 `torch` 与 `torchvision` 必须互相兼容。

安装后先验证：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

CPU 电脑预期输出 `False`。`predict.py` 会自动选择 CPU，无需另加设备参数。

## 6. 命令行预测

在仓库根目录运行。以下命令分别使用原版 v3 和老化微调版：

```powershell
python train/predict.py `
  --checkpoint-path train/experiments/five_band_train/task_specific/background4_v3/background4_v3.pt `
  --five-band-scenes-root train/camera_eval_workspace `
  --output-root train/experiments/five_band_predictions/task_specific/background4_v3 `
  --scene-ids SAMPLE_103 `
  --background-role 代赭 `
  --threshold 0.5 `
  --save-aging-probability-map
```

```powershell
python train/predict.py `
  --checkpoint-path train/experiments/five_band_train/task_specific/background4_v3_agingfix_v1/background4_v3_agingfix_v1_best.pt `
  --five-band-scenes-root train/camera_eval_workspace `
  --output-root train/experiments/five_band_predictions/task_specific/background4_v3_agingfix_v1_best `
  --scene-ids SAMPLE_103 `
  --background-role 代赭 `
  --threshold 0.5 `
  --save-aging-probability-map
```

需要修改的是：

| 目的 | 参数 |
| --- | --- |
| 换模型 | `--checkpoint-path` |
| 换样本 | `--scene-ids` |
| 换样本根目录 | `--five-band-scenes-root` |
| 换输出目录 | `--output-root` |
| 指定真实背景板 | `--background-role` |

一次命令中的样本必须使用同一种真实背景角色。不同背景板应拆成多次预测。背景角色用于提供模型所需的背景条件，不允许把其他真实颜料面积强制压成背景。

## 7. 启动 UI

目标电脑需要 Node.js。启动前在已激活的 Conda 环境中设置：

```powershell
$env:WORKBENCH_PYTHON="$env:CONDA_PREFIX\python.exe"
$env:CAMERA_IMAGE_ROOT="D:\Software\HuaTengVision\Image"
node ui/analysis_workbench/server.mjs D:\multi-optical 8768
```

然后访问：

```text
http://127.0.0.1:8768/ui/analysis_workbench/index.html
```

如果目标电脑的仓库或相机目录不同，只修改实际绝对路径。不要把旧机器的 Conda Python 绝对路径写死到新机器。

## 8. 迁移后验收

依次确认：

1. `predict.py -h` 能正常显示参数。
2. v3 和 `v3_agingfix_v1` 权重均能加载，没有通道数或键名错误。
3. 同一复杂样本在新旧电脑上的输出尺寸、类别文件和主要区域一致。
4. UI 能切换 `background4_v2 / v3 / v3_agingfix_v1`，默认版本符合当前配置。
5. 选择石青、代赭、石绿、朱砂背景时，背景不会整体进入 `paint`。
6. 真实大面积颜料不会仅因与背景角色不同而被裁掉。
7. 如果接收 BMP，完整走通“相机目录 → 五波段样本 → 预测 → UI 查看”。

CPU 与 GPU 的浮点结果可能存在极小差异，但不应出现整块背景被判为颜料、输出头缺失或类别语义变化。出现这类问题时，应优先核对代码版本、权重版本、19 通道特征构造和 `background-role`。
