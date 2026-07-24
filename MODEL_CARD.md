# Model Card

## 项目用途

本项目针对五波段多光谱壁画图像，提供以下像素级输出：

- `paint`：颜料或绘制主体区域；
- `pollution`：外来污染和附着物；
- `aging`：老化、褪变等区域；
- `pigment`：仅在 paint 区域内输出四类颜料。

四类颜料编码固定为：

| 值 | 类别 |
| ---: | --- |
| 0 | ignore / 背景 / 不确定 |
| 1 | 朱砂 |
| 2 | 代赭 |
| 3 | 石青 |
| 4 | 石绿 |

## 输入

原始采集包含五个波段，顺序固定为：

```text
450 / 550 / 600 / 650 / 700 nm
```

历史 softcomp 模型使用由五波段构造的15通道光谱特征。

background4 模型使用：

```text
15通道光谱特征 + 4通道背景one-hot常量图 = 19通道
```

背景类型固定为：

```text
代赭 / 石青 / 石绿 / 朱砂
```

背景类型是条件元数据，不是颜料像素标签。指定某种背景只用于抑制该背景板误报，不能压掉背景板上真实存在的大面积其他颜料。

## 当前模型与保留的历史模型

| 系列 | 模型目录 | 用途 |
| --- | --- | --- |
| background4 v2 | `background4_v2` | 四背景、四颜料像素级基础版本 |
| background4 v3 | `background4_v3` | 在v2基础上加入污染与老化增量样本 |
| background4 v3 agingfix | `background4_v3_agingfix_v1` | 当前老化专项微调版本 |
| softcomp balanced | `retune_9_scene3647_v10_balanced_softcomp_4849_pollution4447_v1` | 历史主softcomp模型 |
| pollution shape | `retune_9_scene3647_pollutionshape_v2` | 历史污染形态专项模型 |
| aging mix | `retune_9_scene3647_agingmix_4849_v1` | 历史老化专项模型 |

## 权重文件校验

| 文件 | SHA-256 |
| --- | --- |
| `background4_v2.pt` | `F95EA0D44A195C844131DA0679E4856031476729DD6A753ABF22EBBB73200A66` |
| `background4_v3.pt` | `90812685E5A39C6D46C0EE080D14223DF4702443CACA14A56A33B59D9F213096` |
| `background4_v3_agingfix_v1_best.pt` | `E865AF26DDA8BE9CAFD463F020C1351131F6996120EB0E9A6CD735E8742BA198` |
| agingmix `vnir_multitask_bootstrap_latest.pt` | `5768823E99A8EE3A5E5738071883B7CA16945EF00DE62091CC7B339C4E41F4EF` |
| pollutionshape `vnir_multitask_bootstrap_latest.pt` | `26B59418ACE91A87FEFDECCF0772A8072186251CB9058209A23C2F09AC57F5BC` |
| softcomp balanced `vnir_multitask_bootstrap_latest.pt` | `2A8E9430F7C7B8967AB3A4FA929106AD12B6861F8C6F73C180041339509894BC` |

前三个 `background4` 模型是当前可切换链路；后三个 softcomp 专项模型只为论文与历史复现保留。模型权重约61MB/个，仓库中必须通过 Git LFS 保存。

## background4 数据口径

- v2冻结基础集：40个场景、1,500个 Patch；
- v3人工复核增量：23个场景、846个 Patch；
- v3训练合计：63个场景、2,346个 Patch；
- `SAMPLE_094–096` 作为外部验证，不进入v3训练；
- 同一采集组的旋转或光照派生数据不得跨训练、验证和测试集合。

具体样本关系与标注状态以 `readme/样本记录规范.md` 为准。

## 训练约束

- 背景板始终不属于 `paint`；
- `pigment > 0` 的像素必须同时满足 `paint > 0`；
- pigment loss 仅在人工确认的 paint 区域内计算；
- 图像、三头 mask 和 pigment 标签必须同步旋转；
- 纯背景样本用于压低背景误报，但不得压过真实颜料正样本；
- aging 与 pollution 是独立像素头，不应由 pigment 输出替代。

## 限制

- 模型只在当前采集设备、波段顺序、背景类型和标注口径下验证；
- 不应把训练 Patch 上的表现等同于外部泛化能力；
- 新光照、相机响应、背景材质或复杂混合颜料可能造成分布偏移；
- UI选择的背景类型必须与真实背景板一致；
- 输出仅用于研究和辅助复核，不能替代文物保护专业判断。
