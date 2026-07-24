# Change Log

本文件记录正式版本的功能、模型和数据口径变化。日期采用 `YYYY-MM-DD`。

## Unreleased

- 准备重新建立精简 Git 仓库。
- 六个正式模型目录统一纳入版本范围，模型权重改用 Git LFS。
- 原始场景、训练 Patch、批量预测结果、UI 生成覆盖图和迁移压缩包不进入普通 Git。
- 新增 Windows/macOS Python 依赖清单。
- 新增模型卡、仓库范围说明和保留权利声明。

## 2026-07-24 — background4_v3_agingfix_v1

- 基于 `background4_v3` 对 aging 头进行专项微调。
- 最佳权重为第4轮候选：
  `background4_v3_agingfix_v1_best.pt`。
- 保持 `paint / pollution / aging / pigment` 像素级输出。
- UI 支持严格按 v2、v3 和 v3 agingfix 版本读取对应预测；没有该版本预测时不借用其他版本结果。
- 完成 CPU 推理/UI 迁移链路与复杂样本验收包。

## 2026-07-23 — background4_v3

- 从 `background4_v2.pt` 初始化并微调10 epoch。
- 冻结基础集为40个场景、1,500个 Patch。
- 增量集为23个场景、846个 Patch。
- 补充污染和老化样本，保留四背景条件与四类像素级颜料输出。

## 2026-07-22 — background4_v2

- 引入代赭、石青、石绿、朱砂四种背景条件。
- 输出三类病害/区域头以及 paint 内四类颜料图。
- 背景板不属于 paint；背景条件用于降低指定背景板的误报。

## 历史 softcomp 链路

- `retune_9_scene3647_v10_balanced_softcomp_4849_pollution4447_v1`
- `retune_9_scene3647_pollutionshape_v2`
- `retune_9_scene3647_agingmix_4849_v1`

这三套目录作为论文与历史主链路模型保留，不与 background4 系列混称。
