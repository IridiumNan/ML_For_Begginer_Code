# CartPole Q-Learning

表格型 Q-learning 求解 `CartPole-v1`，附带一个贝叶斯优化器（Optuna TPE）用于搜索超参数与学习率策略。

状态空间被离散化成表，因此不需要神经网络，纯 CPU 即可训练。

## 环境

```bash
conda activate carport
pip install -r requirements.txt
```

除 `requirements.txt` 已有的 numpy / gymnasium / matplotlib 外，调参还需要：

```bash
pip install optuna scikit-learn pandas
```

`scikit-learn` 只在需要经典「高斯过程 + EI」时才用得上；TPE 路线不需要它。`pandas` 用于导出 `study.trials_dataframe()`。

## 快速开始

```bash
# 训练一个配置（约 2 分钟）
python train.py --strategy invsqrt --alpha 0.8283 367.9 0.001 \
  --gamma 0.865 --epsilon 0.176 --punish -28.5 \
  --discretizer coarse --episodes 30000 --eval-episodes 100

# 播放训练好的策略
python play.py runs/<name>.policy.pkl --episodes 20

# 渲染画面
python play.py runs/<name>.policy.pkl --render --episodes 5
```

## 贝叶斯调参

三阶段，均为 6 进程并行。每个 worker 一次跑一个训练任务（Q-learning 循环是纯 Python，只有进程能真正并行）。

```bash
# 1. 标定：同一配置跑多个种子，测出种子噪声有多大
python tune.py calibrate --workers 6 --episodes 4000 --seeds 5

# 2. 搜索：60 个 trial，每个 3 个种子
python tune.py search --workers 6 --trials 60 --seeds 3 --episodes 4000

# 3. 验证：前 8 名用 5 个种子 + 更大预算复排
python tune.py validate --workers 6 --top 8 --seeds 5 --episodes 15000
```

`search` 的 study 存在 `tuning/cartpole.db`，中断后重跑会自动续跑。`validate` 最后会产出可直接播放的 `tuning/cartpole_best.policy.pkl`。

**先跑 calibrate 再跑 search。** 实测同一配置换种子，贪心评估相差 30–67 分。单种子 trial 的排名基本是噪声，所以 objective 用多随机种子的均值，最终排名交给验证阶段。

## 学习率策略

`model.py` 里的 `ALPHA_STRATEGIES` 注册了五种，参数按声明顺序传给 `--alpha`：

- **constant** — `alpha`
- **linear** — `a0 a1 horizon`，从 `a0` 线性退火到 `a1`
- **exp** — `a0 decay floor`，几何衰减
- **invsqrt** — `a0 scale floor`，`a0 / sqrt(1 + n/scale)`
- **step** — `a0 factor every floor`，每 `every` 轮乘一次 `factor`

所有策略返回的学习率都会被夹到 `[1e-6, 1.0]`，宽范围搜索不会把 Q 表炸掉。

## 文件

- `train.py` — 训练 + 贪心评估 + 命令行入口。`train()` 可重入：每次调用自带局部 Q 表和独立 RNG
- `model.py` — `Config`、学习率策略、`Result`、策略存取（`save_policy` / `load_policy`）
- `tune.py` — Optuna 三阶段调参 + 进程池
- `play.py` — 加载策略并播放
- `utils.py` — 离散化方案、查表策略
- `convolve.py` — 滑动平均小工具

产物在 `runs/`（策略包、曲线图、JSON）与 `tuning/`（study 数据库、trial 明细 CSV、验证结果）。两者都已加入 `.gitignore`。

## 注意：离散化方案必须匹配

有两套状态网格：

- `fine` — `discretize`，按固定比例缩放取整
- `coarse` — `discretize_bins`，`np.digitize` 分箱，20×20×10×10

**查表用错网格是静默失败**：所有 key 都查不到，两个动作的 Q 值都读成 0，归一化后变成均匀分布，策略退化成纯随机（约 20 分，刚好是随机基线）。

所以策略一律存成自描述的 `*.policy.pkl`，里面记录了 `Q` 表、`Config` 和网格名字，加载时自动还原。旧的裸 `*.plk` 没有元数据，需要手动指定：

```bash
python play.py runs/old.plk --discretizer coarse --episodes 20

# 不确定是哪个网格时，让它自己试
python play.py runs/old.plk --sweep
```

`--sweep` 会把两套网格都跑一遍，分高的那行就是它真正的网格。

## 目前的结果

60 个 trial 搜索 + 前 8 名 × 5 种子 × 15000 episodes 验证，最优配置：

``` 
strategy  invsqrt   alpha(n) = 0.8283 / sqrt(1 + n/367.9)
gamma     0.865
epsilon   0.176
punish    -28.5
grid      coarse

贪心评估  mean 353.31   sd 75.10   (5 种子)
```

搜出来的策略有一个共同规律：`linear` 和 `step` 全部被淘汰出前列，`invsqrt` / `constant` / `exp` 占优，且排名靠前者一致落在高 `epsilon`（0.15–0.29）与 `gamma ≈ 0.87–0.95`。在现有离散化粒度下，欠探索比过探索更致命。

把预算加到 30000 episodes 后，贪心评估可到 **约 370–400**，单局能跑满 500。

**天花板说明**：这套表格方法的瓶颈是状态离散化粒度，不是超参数。继续调参的边际收益很低，想稳定拿满分需要换函数逼近（如 tile coding 或小型神经网络）。
