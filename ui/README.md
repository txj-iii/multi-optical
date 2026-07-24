# UI Analysis Workbench

当前工作台目录：

```text
D:\multi-optical\ui\analysis_workbench
```

`D:\multi-optical\1` 是旧备份，不是当前UI入口。

## 功能

- 查看 `preview.png`
- 查看 paint / pollution / aging / pigment预测
- 切换background4版本
- 保存样本背景板
- 运行softcomp起标或background4预测
- 查看和修改人工masks
- 保存、采用或暂不采用标注
- 查看五波段曲线和颜料子区分析

## 两类预测链

### Softcomp review

```text
validation_v10_balanced_softcomp_5056_pollthr035
```

用途：

- 论文version 2结果
- UI自动起标
- 生成可编辑标注草稿

### Background4

下拉框支持：

- `background4_v2`：默认
- `background4_v3`
- `background4_v3_agingfix_v1_best`

background4使用19通道背景条件模型。运行前必须选择并保存：

```text
代赭 / 石青 / 石绿 / 朱砂
```

背景板只用于模型条件输入，不能自动当作颜料标签。

## UI工作流

```text
导入最新5张BMP
-> 生成five_band.npy和preview.png
-> 选择背景板
-> 运行预测
-> 查看标注草稿
-> 人工修改并保存
-> 采用或暂不采用
```

规则：

- “保存标注”只更新当前样本masks。
- “采用标注”只更新审核状态。
- 保存或采用都不会自动训练。
- 已人工保存的masks不能被预测草稿静默覆盖。
- 不同版本预测写入独立目录。

## 主要文件

```text
ui/analysis_workbench/
├── index.html
├── styles.css
├── app.js
├── server.mjs
├── workflow_state.json
├── workbench_manifest.json
├── workbench_manifest.js
└── workbench_version_provenance.json
```

后端：

```text
train/workbench_workflow.py
train/analysis_workbench.py
train/predict.py
```

## 状态与manifest

- `workflow_state.json`：样本状态、背景板、光照、方向和当前预测版本。
- `workbench_manifest.json`：UI当前样本与资产清单。
- `workbench_manifest.js`：静态兼容副本。
- `workbench_version_provenance.json`：版本来源说明。

迁移UI时不要用空状态文件直接覆盖目标电脑已有 `workflow_state.json`。

## 启动

先激活安装了PyTorch的Conda环境：

```powershell
conda activate pipe2
$env:WORKBENCH_PYTHON="$env:CONDA_PREFIX\python.exe"
```

相机目录不在默认位置时：

```powershell
$env:CAMERA_IMAGE_ROOT="你的相机BMP目录"
```

启动：

```powershell
node D:\multi-optical\ui\analysis_workbench\server.mjs D:\multi-optical 8768
```

访问：

```text
http://127.0.0.1:8768/ui/analysis_workbench/index.html
```

## 当前注意事项

- `SAMPLE_094–096` 无确认人工真值且背景记录冲突，不作为正式验证结论。
- background4预测缺少背景板时应直接报错。
- 页面显示旧结果时，先确认版本下拉框，再重新运行该版本预测。
- 同一版本同一样本重新预测会覆盖该预测目录中的同名文件，但不会覆盖人工masks。

完整训练和版本规则见：

- [训练学习规则.md](D:/multi-optical/readme/训练学习规则.md)
- [样本记录规范.md](D:/multi-optical/readme/样本记录规范.md)
