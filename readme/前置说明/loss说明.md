# loss 说明

本文只说明当前 `background4` 联合训练的 loss 口径。更早的三头、patch 级 pigment 实验属于历史链路。

## 三个病害头

`paint / pollution / aging` 都是像素级二分类任务。每个 head 的基础损失由 BCE 与 Dice 组合，再按任务权重汇总。

对应真值：

| 输出 | 真值 |
| --- | --- |
| `paint` | `paint.png` |
| `pollution` | `pollution.png` |
| `aging` | `aging.png` |

任务权重支持固定等权和 DWA。具体参数与实际记录以本次训练目录中的配置和 `training_log.txt` 为准，不能只根据某份旧文档还原。

## pigment 头

当前 `background4` 的 `pigment` 是像素级四分类辅助任务：

- 1：朱砂
- 2：代赭
- 3：石青
- 4：石绿
- 0：背景、不确定或 ignore

pigment loss 只在人工确认且位于 paint 内的有效像素上计算。必须满足：

```text
pigment > 0  ⇒  paint > 0
```

预测 pigment 图可以辅助人工标注，但不能直接作为真值参与训练。

## 训练与评价的区别

loss 是优化目标，不等于最终质量。训练完成后仍需在按场景隔离的验证/测试集上分别计算：

- IoU
- Dice
- Precision
- Recall

当前还要专项检查石青背景上的代赭召回、大面积颜料召回，以及石青纯背景老化的 aging 召回和 paint 误报面积。统一评价口径见 [三分类结果验证与评价指标说明](../三分类结果验证与评价指标说明.md)。
