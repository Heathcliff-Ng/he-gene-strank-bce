# 模型与实验口径

项目中的 STRank+BCE 将多项计数建模用于同一细胞内的跨基因排序，并用独立的 BCE 头判断检出。历史配置名为 `multinomial_bce`，周报的对比方法名为 `original_count_bce`。

每个细胞输入 1024 维冻结 H&E UNI 特征，按源训练分区的逐维均值和标准差做标准化。共享主干为 Linear(1024, 256)、GELU、dropout 0.1。两个线性头各输出 313 个 logits。计数头输出相对排序分数 r，检出头输出 a，取 sigmoid 后用于选集。权重同时保存标准化参数、面板顺序和训练时的阈值。[模型](../src/he_gene_strank_bce/model.py) · [推理](../src/he_gene_strank_bce/predict.py)

### detected 训练

计数损失为 `mean_cells(−sum_genes(y × log_softmax(r))) / mean_total_train`。每个细胞的原始计数直接作为权重。归一化常数是源训练分区的每细胞平均面板总计数。检出损失为所有细胞、所有面板基因上 `BCEWithLogits(a, y > 0)` 的均值。两项相加，BCE 系数为 1。计数为 1 时两个头都收到监督。用源片空间验证集宏平均 AP 选择 epoch，再在该验证集上从 0 到 1 的 101 个等距阈值中选择宏平均 F₁.₂ 最高的工作点。[联合损失](../src/he_gene_strank_bce/losses.py) · [训练](../src/he_gene_strank_bce/train.py)

### neutral 冻结复评

保持同一权重，检出头仍按原训练定义打分，改变的是决策阈值和评价标签。用来自源片原空间验证区的 1,024 个校准细胞，穷举分数阈值（含全不选）并最小化 FP + 2FN。相同分数同时入选，成本并列优先 FP 较少，再优先总入选数较少；随后把阈值冻结用于目标片。测试时真实计数 1 不影响 FP/FN，入选的 1 保留在预测集合中。历史跨片评价使用每片固定的 4,096 个测试细胞。[协议实现](../src/he_gene_strank_bce/protocols.py) · [校准和测试细胞清单](../data/cohorts/)

### neutral 从头训练

设置 `train --singletons neutral` 后，训练使用同一 MLP，并对两个头屏蔽真实计数 1。计数头只在有效基因间计算 log-softmax，源训练分区的平均总计数也只统计保留计数。BCE 以 ≥2 为阳性、0 为阴性，每个细胞先对有效位求平均，再对有监督的细胞平均。整行都是 1 时不提供监督。输出偏置的初始化使用保留计数及有效检出标签。每轮在源片空间验证区以最低可达 FP + 2FN 选模，所选 epoch 再用同一区域定阈值。这个选项生成新的权重和结果，目录中的六份冻结权重均来自 detected 训练。

早期项目另有 `masked_count1_20261020` 的 A/B 匹配重训，采用冷初始化、dropout 0 和困难区域选模等设置。若引用本目录的 [历史结果](../reports/results.json)，其 neutral 项对应上面的冻结复评。
