# GitHub仓库范围与上传准备

## 目标

新仓库保存可维护、可复现的当前代码版本，同时避免原始图像、训练Patch和批量预测结果再次把 Git 历史膨胀到数GB。

本轮只准备文件，不执行 `git add`、提交或上传。

## 纳入范围

### 源代码

```text
train/*.py
train/tests/*.py
train/README.md
ui/README.md
ui/analysis_workbench/*.html
ui/analysis_workbench/*.css
ui/analysis_workbench/*.js
ui/analysis_workbench/*.mjs
ui/analysis_workbench/*.json
```

UI运行时生成的 manifest、workflow state 和覆盖图不纳入。

### 文档与环境

```text
README.md
readme/
requirements.txt
requirements-macos.txt
LICENSE.md
CHANGELOG.md
MODEL_CARD.md
.gitignore
.gitattributes
```

### 六个正式模型目录

```text
train/experiments/five_band_train/task_specific/background4_v2/
train/experiments/five_band_train/task_specific/background4_v3/
train/experiments/five_band_train/task_specific/background4_v3_agingfix_v1/
train/experiments/five_band_train/task_specific/retune_9_scene3647_agingmix_4849_v1/
train/experiments/five_band_train/task_specific/retune_9_scene3647_pollutionshape_v2/
train/experiments/five_band_train/task_specific/retune_9_scene3647_v10_balanced_softcomp_4849_pollution4447_v1/
```

六个目录中的 `.pt` 权重由 Git LFS 管理，训练日志作为普通文本管理。

## 不纳入范围

```text
train/camera_eval_workspace/
train/five_band_patches/
train/experiments/five_band_predictions/
ui/analysis_workbench/generated_overlays/
ui/analysis_workbench/workflow_state.json
ui/analysis_workbench/workbench_manifest.js
ui/analysis_workbench/workbench_manifest.json
outputs/
ppt/
experiment_figures/
.venv/
.idea/
*.zip
*.npy
*.npz
```

这些内容应在本地硬盘或独立数据归档中保存。人工标注与原始五波段场景不能只依赖 GitHub。

## Patch清单

实际 Patch 不进入 Git。为了记录训练数据快照，以下小型清单已经复制到新的 `train/manifests/`：

```text
background4_v1/training_manifest.json
background4_v1/protected_softcomp_baseline.json
background4_v1/train/patch_index.csv
background4_v3/training_manifest.json
background4_v3/train/patch_index.csv
```

当前可提交目录为：

```text
train/manifests/background4_v2/
train/manifests/background4_v3/
```

只提交这些清单副本，不提交 `images/`、`masks/` 或 `.npy`。

## Git LFS准备

新电脑和当前电脑都需要安装 Git LFS：

```powershell
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
```

仓库已经通过 `.gitattributes` 声明 `.pt/.pth` 使用 LFS。提交前必须检查：

```powershell
git lfs ls-files
git check-ignore -v train/experiments/five_band_train/task_specific/background4_v3_agingfix_v1/background4_v3_agingfix_v1_best.pt
```

第二条命令不应显示该模型被 `.gitignore` 排除。

## 正式上传前检查

1. 备份当前 `.git`，不要先直接删除。
2. 确认新 `.gitignore` 生效。
3. 安装并初始化 Git LFS。
4. 检查六个模型权重都显示为 LFS 文件。
5. 检查 `git status` 中没有场景、Patch、预测图、覆盖图或 ZIP。
6. 运行 Python 测试和 UI JavaScript 测试。
7. 首次提交后检查 GitHub 页面中的模型显示为 Git LFS 指针管理。
8. 新位置完整 clone 一次，验证模型能下载、UI能启动、推理能读取 checkpoint。

## 当前注意事项

当前旧 `.git` 约6.23GB，包含历史大型对象。仅修改 `.gitignore` 不会缩小旧历史。准备完成并确认新仓库内容正确后，应采用“备份旧 `.git` → 初始化干净仓库 → 首次提交”的方式迁移，而不是在旧历史上继续提交。
