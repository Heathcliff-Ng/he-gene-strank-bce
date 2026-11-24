# 训练和复评

从独立目录根目录运行。准备两片 Xenium H5AD，默认路径为 `data/Rep1.h5ad`、`data/Rep2.h5ad`，也可以传入已有文件的绝对路径。H5AD 使用 `layers/raw` 的非负整数计数、`obsm/he` 的冻结浮点特征、`obsm/image_coor` 的两列空间坐标，以及 obs 细胞索引、var 基因名称索引、`var/gene_ids` 和 `var/feature_types`。训练按稳定的基因 ID 对齐两片，历史权重要求同一 313 基因面板及同种 UNI 特征提取方式。

在原 Rank 工作区里可直接指向已有文件：

```bash
REP1=../datasets/Human_Breast_Cancer_Rep1/Human_Breast_Cancer_Rep1_uni_resolution64_full_with_spatialex_all_pred.h5ad
REP2=../datasets/Human_Breast_Cancer_Rep2/Human_Breast_Cancer_Rep2_uni_resolution64_full_with_spatialex_all_pred.h5ad
CKPT=checkpoints/Rep1/seed_42.pt
```

工作区外可将 `REP1`、`REP2` 设置为自己的文件。下文命令假设已在当前 Python 环境执行 `python -m pip install -e .`。也可以从目录根使用 `PYTHONPATH=src python -m he_gene_strank_bce` 替代 `he-gene-strank-bce`。

## 直接复评已有权重

以下两个命令分别重现 Rep1 → Rep2、seed 42 的原始检出评价和周报中性评价。

```bash
he-gene-strank-bce evaluate --protocol detected --checkpoint "$CKPT" \
  --section Rep2 --data "$REP2" --output runs/detected_rep2_seed42.json

he-gene-strank-bce evaluate --protocol neutral --checkpoint "$CKPT" \
  --section Rep2 --data "$REP2" --ids data/cohorts/Rep2_test.txt \
  --calibration reports/checkpoints.json --output runs/neutral_rep2_seed42.json
```

把 `seed_42.pt` 换为 `seed_43.pt`、`seed_44.pt` 可得到其余重复。反向实验选择 `checkpoints/Rep2/` 的权重、`Rep1` 的切片及相应测试 ID。源片、方向、阈值和种子记录都可在[结果清单](../reports/results.json)核对。不同设备的浮点舍入可能影响恰好位于阈值的预测。

## 重新校准得到中性阈值

```bash
he-gene-strank-bce calibrate --protocol neutral --checkpoint "$CKPT" \
  --section Rep1 --data "$REP1" --ids data/cohorts/Rep1_calibration.txt \
  --output runs/calibration_rep1_seed42.json
```

复评或推理时将 `--calibration` 指向这个 JSON 即可。校准使用模型的源切片，指定的 ID 必须属于源空间验证区。权重 SHA256 绑定校准结果；历史冻结清单也按此校验值匹配权重。新数据自行划分源训练、验证与目标测试，随附的 ID 只适用于原始两片。

## 从头训练

两条命令分别生成自己的模型与报告。默认训练两个源片，每个源片使用 42、43、44 三个种子。如只需单个组合，可加 `--source Rep1 --seed 42`。

```bash
he-gene-strank-bce train --config configs/training.yaml --singletons detected \
  --rep1 "$REP1" --rep2 "$REP2" --output-dir runs/train_detected

he-gene-strank-bce train --config configs/training.yaml --singletons neutral \
  --rep1 "$REP1" --rep2 "$REP2" --output-dir runs/train_neutral
```

同一份配置固定网络和优化器。每个源片沿图像坐标第 0 维留出 20% 的空间尾部，并设置 64 像素缓冲。AdamW 学习率 0.001，weight decay 0.0001，batch 2048，最多 40 epoch，patience 8。各训练口径的损失、选模和阈值规则见[方法说明](method.md)。

一次运行产出 `<output-dir>/<方向>/seed_<种子>/` 下的 `validation_model.pt`、`split.npz`、`config.json` 和 `report.json`。推理/评价新训练的模型时，匹配训练口径可直接使用检查点内的阈值。对新权重另定阈值，请使用该权重重新校准。

## 本地验证

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

测试用临时合成数据覆盖两种训练口径、单计数位点梯度、源片阈值与仅图像部署，并核对随目录附带的六份权重。

## 真实数据复核（2026-11-24）

复核使用 NVIDIA GeForce RTX 5070 Ti、Python 3.10.13、PyTorch 2.11.0+cu128、CUDA 12.8、NumPy 1.26.4、SciPy 1.13.1 和 h5py 3.11.0。输入为同一乳腺癌标本的两张 Xenium 切片；在原始数据所在目录对完整 H5AD 运行 `sha256sum` 可核对版本。

| 输入文件 | SHA256 |
|---|---|
| `Human_Breast_Cancer_Rep1_uni_resolution64_full_with_spatialex_all_pred.h5ad` | `e43e21fd001e6a20be38c53d832f6389ca1f07a9369c5bffd212b680f1b24327` |
| `Human_Breast_Cancer_Rep2_uni_resolution64_full_with_spatialex_all_pred.h5ad` | `b26c2f7344314135a61a2c1bdf5021474794ebb01c43f5962b0413476665d346` |

**从头训练。** 两个训练方向各跑 seed 42、43、44，均采用 `detected`。逐一比较切分中的细胞 ID、逐轮训练损失、验证 AP、所选 epoch、阈值和检查点。121 轮损失和验证 AP 全部相同。六个 `state_dict` 中的 48 个张量（含特征标准化统计，共 2,551,980 个数值）与原权重逐位相同。检查点文件还保存本次运行的元数据，复核重训权重时应比较其中的张量。重训命令加了 `--skip-kendall`，Kendall 在随后独立加载权重时完整计算。

| 训练来源 | seed 42 最优轮次 / 阈值 | seed 43 最优轮次 / 阈值 | seed 44 最优轮次 / 阈值 |
|---|---|---|---|
| Rep1 | 15 / 0.22 | 19 / 0.23 | 13 / 0.22 |
| Rep2 | 12 / 0.22 | 7 / 0.22 | 7 / 0.23 |

**冻结权重复评。** 对每个种子，Rep1 → Rep2 评价全部 111,555 个目标细胞，Rep2 → Rep1 评价全部 164,000 个目标细胞。`detected` 的逐种子 AP、阳性基因 Spearman、Kendall 和选集指标均复现；六份跨片报告各 197 个数值逐项对齐。由训练命令直接汇总的 NDCG 与历史报告最多相差 `2.22e−16`。

**周报中性口径。** 每份冻结权重分别在源片 1,024 个细胞上重新校准，六个阈值完全复现。每个方向、每个种子取目标片 4,096 个细胞，合计 7,692,288 个细胞－基因入选判断均与历史归档相同。Rep1 → Rep2 和 Rep2 → Rep1 的每细胞平均 FP + 2FN 分别为 42.455729 和 42.158529。以上数字属于原模型的中性复评。`train --singletons neutral` 会产生另一次实验的权重。
