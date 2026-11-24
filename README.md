# HE-Gene-STRank-BCE

从单细胞 H&E 的冻结 UNI 特征同时预测 Xenium 面板基因的检出概率和细胞内计数排序。这个目录可以单独使用。它包含模型、训练和推理代码，两个切片来源各三个种子的冻结权重，以及两种 count=1 口径的历史阈值、测试细胞 ID 和结果。

## 两种实验方式

| | `detected` · count=1 算检出 | `neutral` · count=1 算中性 |
|---|---|---|
| 评价标签 | 0 为阴性，≥1 为阳性 | 0 为阴性，≥2 为阳性，1 不计 TP、FP 或 FN |
| 周报所用权重 | 以 `y>0` 监督检出头，以全部原始计数监督计数头 | 复用左列的六份权重 |
| 历史阈值 | 源片空间验证集上最大化逐细胞宏平均 F₁.₂，阈值随权重保存 | 每份权重用源片 1,024 个校准细胞重新选阈值，使每细胞平均 FP + 2FN 最小 |
| 历史跨片评价范围 | 完整目标片 | 固定的 4,096 个目标片测试细胞 |
| 预测集合 | 选择检出概率 ≥ 阈值的基因，按计数分数排序 | 同一选择规则；若选中一个真实计数为 1 的基因，它计入集合大小，也单列为 `selected_ignored` |

若要**从训练时就屏蔽 count=1**，训练命令也接受 `--singletons neutral`。这会学习新的模型参数。上表的历史中性结果来自冻结原模型后的复评。两种做法的训练监督与校准样本见[方法说明](docs/method.md)。

## 快速使用

在本目录执行安装。训练和评价需要原始 H5AD。仅做推理时，只需要每个细胞的 UNI 特征。冻结权重与阈值清单已随目录提供。

```bash
python -m pip install -e .
he-gene-strank-bce --help
```

下面对同一份特征做两种预测。将 `FEATURES` 指向含 `obsm/he` 的 H5AD，或含 `features`、`cell_ids` 的 NPZ。

```bash
FEATURES=data/Rep2.h5ad
CKPT=checkpoints/Rep1/seed_42.pt

he-gene-strank-bce predict --checkpoint "$CKPT" --section Rep2 --features "$FEATURES" \
  --protocol detected --output runs/rep2_detected.npz
he-gene-strank-bce predict --checkpoint "$CKPT" --section Rep2 --features "$FEATURES" \
  --protocol neutral --calibration reports/checkpoints.json --output runs/rep2_neutral.npz
```

输出的 `detection_probabilities` 和 `ranking_scores` 分别是两头的分数。`selected` 是布尔选集，`selected_order` 保存入选基因在面板中的列索引，末尾用 −1 填充。两次调用使用同一权重，因此改变的是阈值及选集。新样本推理无需提供 RNA 或每细胞的真实计数总量。

## 历史结果

两个方向来自同一标本。每个数是三个种子的均值，两列的测试细胞范围和监督定义见上表。

| 跨片方向 | detected 检出 AP | detected 阳性基因内 Spearman | neutral 每细胞 FP | neutral 每细胞 FN | neutral 每细胞 FP + 2FN |
|---|---:|---:|---:|---:|---:|
| Rep1 → Rep2 | 0.6578 | 0.5057 | 14.0701 | 14.1928 | 42.4557 |
| Rep2 → Rep1 | 0.6622 | 0.5158 | 13.4985 | 14.3300 | 42.1585 |

逐种子的指标保存在[实验结果](reports/results.json)，阈值、权重 SHA256 和细胞清单校验值保存在[冻结记录](reports/checkpoints.json)。

## 目录入口

| 任务 | 位置 |
|---|---|
| 了解结构、损失和 count=1 的训练语义 | [方法说明](docs/method.md) |
| 准备数据、从头训练、重新校准、复评已有权重 | [复现命令](docs/reproduction.md) · [默认配置](configs/training.yaml) |
| 查找训练、推理和评价实现 | [源码](src/he_gene_strank_bce/) |
| 选择一个源片和种子 | [模型权重](checkpoints/) · [历史测试和校准 ID](data/cohorts/) |
| 检查训练掩码、阈值和端到端流程 | [测试](tests/) |
