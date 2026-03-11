# A股 5 分钟 PPO + Tiny Transformer 训练框架（执行版草案）

> 目标：采用“路线 2 + 路线 3”（轻量 Transformer + 加速），统一配置，支持多股票随机采样训练，并保留最佳模型。

## 1. 重构目标

- 股票池、采样参数、模型结构、加速开关统一放在 `config.json`。
- `train.py` 与 `main.py` 只负责流程调度，不再硬编码参数。
- 代码风格保持短小、结构化、可读；减少冗余 fallback。
- 出错直接抛错，避免静默吞错。

---

## 2. 数据与采样规则

### 2.1 交易时间与 bar

- A 股每天 4 小时交易时间。
- 使用 5 分钟 bar，则每天 `48` 根 bar。

### 2.2 训练/推理/交易股票划分

- `train_codes`：训练股票代码列表。
- `infer_codes`：推理股票代码列表（通常为未参与训练的股票）。
- `trade_codes`：交易股票列表（对象结构：`code` + `name`）。
- 每个股票数据长度可不同。
- `topK` 在实盘 `main.py` 中表示“从 `trade_codes` 候选池最多管理多少只交易标的（槽位数）”，不是按评分截断候选池。
- `run_full_infer=true` 时，`main.py` 在启动和跨日会对 `train_codes + infer_codes + trade_codes` 做一次“空仓推理”，并按分数降序打印。

### 2.3 有效采样区间

每个股票维护统一逻辑的有效区间（训练）：

- `valid_start` 由“窗口完整性 + 指标预热”共同决定（均来自 `config.json`）：
  - `sampler.infer_window_bars`（一次 infer/Transformer 输入窗口长度）；
  - `sampler.indicator_warmup_bars`（指标预热条数，如 `macd_5: 4`）；
  - `sampler.extra_warmup_bars`（额外预热冗余，默认可为 `0`）。
- `valid_end` 由 `sampler.rollout_steps` 与 `sampler.next_open_guard_bars` 决定，保证 rollout 不越界且下一根开盘价可用。
- 训练时随机选：
  - 一个训练股票；
  - 该股票 `[valid_start, valid_end]` 内的随机起点。

> 训练采样公式：`warmup_bars = max(max(indicator_warmup_bars.values()), extra_warmup_bars)`；`valid_start = (infer_window_bars - 1) + warmup_bars`；`valid_end = data_len - rollout_steps - next_open_guard_bars - 1`。若 `valid_end <= valid_start`，该股票本轮跳过。
>
> 实盘/离线单步推理（最新 bar）不使用上面的 `valid_end`，而是：
> `infer_start_idx = data_len - next_open_guard_bars - 1`，并要求 `infer_start_idx > valid_start`。
> 这样不会被训练 `rollout_steps`（如 4096）错误限制，避免 `_infer.csv` 因行数较短而无法推理。

---

## 3. 环境定义

### 3.1 行情数据

- 使用后复权 K 线。
- 不做归一化，不依赖 `.pkl` 标准化文件。

### 3.2 初始账户

- 初始总现金：`100000`
- 初始总资产：`100000`

### 3.3 手续费

- 买入手续费：万 1（`0.0001`）
- 卖出手续费：万 6（万 1 + 印花税万 5，合计 `0.0006`）

### 3.4 持仓与交易约束

- A 股整手交易：100 股整数倍。
- T+1：当日买入部分记为锁定股数，不可当日卖出。

### 3.5 观测（obs）

每个 bar 输入至少包含：

1. 当前 bar 收盘价（用于盯市与特征，不作为当前步撮合价）
2. 总现金
3. 总资产
4. 可买股数（100 股倍数）
5. 可卖股数（100 股倍数）
6. 锁定股数（T+1）
7. 当前持仓成本价
8. 技术指标特征（由 `features.indicators` 配置）

推荐起步指标：

`INDICATORS = ["macd", "rsi_1440", "cci_1440", "dx_1440", "close_1440_sma", "close_2880_sma", "volume_240_sma", "atr_240"]`

### 3.6 动作（action）

- 动作维度：`1`
- 取值范围：`[-1, 1]`
  - `a > 0`：买入比例 = `a`
  - `a < 0`：卖出比例 = `abs(a)`
- 决策与成交时序：
  - 在 `bar_t` 收盘后做决策（可看到最新 5 分钟 bar）；
  - 在 `bar_{t+1}` 开盘价执行买卖撮合。
- 实际下单股数：
  - 买：`floor(可买股数 * a / 100) * 100`
  - 卖：`floor(可卖股数 * abs(a) / 100) * 100`

> 说明：对 PPO 连续动作空间是可行的，梯度由策略分布参数学习，不依赖环境中的取整操作反传。

### 3.7 奖励（reward）

- 盯市口径：市值按“当前决策时点可见的最新 bar 收盘价”计算。
- 撮合口径：动作在下一根 bar 开盘价执行。
- reward 建议定义为：`equity_close_{t+1} - equity_close_t`（含 `open_{t+1}` 执行后的持仓变化影响）。

---

## 4. 训练循环

- 一次 PPO 数据收集 = 一个 rollout buffer（`n_steps`）。
- 每次 collect 后：
  1. 执行一次 PPO update；
  2. 重新随机股票与随机起点；
  3. 循环直到训练结束。
- 约束：`ppo.n_steps` 与 `sampler.rollout_steps` 保持一致。
- `rollout_steps`、`infer_window_bars`、`indicator_warmup_bars`、`features.indicators`、PPO 超参数均在 `config.json` 可调。

---

## 5. 验证与模型保存

### 5.1 验证触发

- 每收集 `train.validate_every_rollouts` 个 rollout 后进行一次验证。

### 5.2 验证方式

- 对 `infer_codes` 中每个股票：
  - 从该股票第一条有效数据开始；
  - 跑到最后一条数据；
  - 计算单标的累计收益。
- 汇总得到验证总收益。

### 5.3 模型保留策略

- 仅保留一个文件名：`best_model`（避免文件爆炸）。
- 若本次验证更优，覆盖保存 `best_model`。
- 同时追加日志，记录完整信息：
  - 总收益
  - 各验证股票分收益（到个位即可）
  - 触发时 rollout 序号 / step
  - 时间戳

---

## 6. 模型路线（采用路线 2 + 路线 3）

### 6.1 Tiny Transformer Feature Extractor（轻量）

- 使用 SB3 自定义特征提取器：`TinyTransformerFeatureExtractor`。
- 采用 `decoder_only` 架构。
- `window_size`（一次性输入的 K 线条数）可配置。
- `n_layers`、`d_model`、`n_heads`、`ffn_dim`、`dropout` 均可配置。
- 推荐默认规模（先跑通再调大）：
  - `d_model = 64`
  - `n_heads = 4`
  - `n_layers = 2`
  - `ffn_dim = 128`
  - `dropout = 0.1`
- 提取器只负责时序特征编码，输出固定维度向量给 PPO 的 actor/critic 头。

### 6.2 SB3 自定义 policy（官方支持）

- 用 `policy_kwargs` 挂自定义 `features_extractor_class` 与 `features_extractor_kwargs`。
- policy/value 头保持小网络，建议：
  - `pi = [64, 64]`
  - `vf = [64, 64]`

### 6.3 路线 3：进一步加速

- 注意力实现使用 `scaled_dot_product_attention`（SDPA）。
- 启用 `torch.compile()`：
  - `mode = "reduce-overhead"`
  - `fullgraph = false`
  - `dynamic = false`
- 保留配置开关，遇到驱动/算子兼容性问题可快速回退。

---

## 7. 统一配置文件示例（`config.json`）

```json
{
  "topK": 5,
  "run_full_infer": true,
  "train_codes": [
    "sh.600018",
    "sh.600025",
    "sh.600027",
    "sh.600062",
    "sh.600098",
    "sh.600233",
    "sh.600269",
    "sh.600398",
    "sh.600517",
    "sh.600572",
    "sh.600642",
    "sh.600690",
    "sh.600710",
    "sh.600873",
    "sh.600887",
    "sh.600926",
    "sh.600970",
    "sh.601918",
    "sh.601965",
    "sh.601991",
    "sh.603558",
    "sh.603600",
    "sz.000338",
    "sz.000404",
    "sz.000528",
    "sz.000543",
    "sz.000729",
    "sz.000899",
    "sz.002034",
    "sz.002078",
    "sz.002116",
    "sz.002236",
    "sz.002588",
    "sz.002832"
  ],
  "infer_codes": [
    "sh.600023",
    "sh.600039",
    "sh.600011"
  ],
  "trade_codes": [
    { "code": "sz.000543", "name": "皖能电力" },
    { "code": "sh.603600", "name": "永艺股份" },
    { "code": "sh.600269", "name": "赣粤高速" },
    { "code": "sh.600011", "name": "华能国际" },
    { "code": "sh.600572", "name": "康恩贝" }
  ],
  "features": {
    "indicators": [
      "macd",
      "rsi_1440",
      "cci_1440",
      "dx_1440",
      "close_1440_sma",
      "close_2880_sma",
      "volume_240_sma",
      "atr_240"
    ]
  },
  "sampler": {
    "rollout_steps": 4096,
    "infer_window_bars": 240,
    "indicator_warmup_bars": {
      "macd": 33,
      "macd_5": 4,
      "rsi_1440": 1439,
      "cci_1440": 1439,
      "dx_1440": 1439,
      "close_1440_sma": 1439,
      "close_2880_sma": 2879,
      "volume_240_sma": 239,
      "atr_240": 239
    },
    "extra_warmup_bars": 0,
    "next_open_guard_bars": 1,
    "valid_start_formula": "(infer_window_bars - 1) + max(max(indicator_warmup_bars.values()), extra_warmup_bars)",
    "valid_end_formula": "data_len - rollout_steps - next_open_guard_bars - 1"
  },
  "model": {
    "mode": "tiny_transformer",
    "features_extractor": {
      "name": "TinyTransformerFeatureExtractor",
      "architecture": "decoder_only",
      "window_size": 240,
      "d_model": 64,
      "n_heads": 4,
      "n_layers": 2,
      "ffn_dim": 128,
      "dropout": 0.1,
      "pooling": "last"
    },
    "policy_head": {
      "pi": [64, 64],
      "vf": [64, 64]
    },
    "acceleration": {
      "use_sdpa": true,
      "use_torch_compile": true,
      "torch_compile_mode": "reduce-overhead",
      "torch_compile_fullgraph": false,
      "torch_compile_dynamic": false
    }
  },
  "ppo": {
    "learning_rate": 0.0003,
    "n_steps": 4096,
    "batch_size": 512,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01
  },
  "train": {
    "total_timesteps": 5000000,
    "validate_every_rollouts": 1000,
    "device": "cuda"
  },
  "trade": {
    "min_fee": 5.0,
    "open_full_threshold": 0.7,
    "flat_threshold": 0.3,
    "open_candidate_threshold": 0.3,
    "rebalance_diff_threshold": 0.1,
    "min_trade_value": 100.0,
    "bar_interval_minutes": 5,
    "order_settle_initial_wait_seconds": 3,
    "order_settle_max_wait_seconds": 10,
    "in_bar_poll_sleep_seconds": 0.1,
    "out_of_trading_sleep_seconds": 1.0,
    "xiadan_path": "C:\\\\同花顺软件\\\\同花顺\\\\xiadan.exe",
    "trading_sessions": [
      ["09:35:00", "11:30:00"],
      ["13:00:00", "15:00:00"]
    ]
  }
}
```

---

## 8. 代码结构建议

- `config.py`：加载与校验 `config.json`。
- `features.py`：指标计算与 `indicator_warmup_bars` 管理。
- `extractor_tiny_transformer.py`：`TinyTransformerFeatureExtractor`（含 SDPA）。
- `policy.py`：SB3 `policy_kwargs` 组装与自定义 policy 注册。
- `sampler.py`：依据 `rollout_steps`、`infer_window_bars`、指标预热计算有效区间并采样。
- `data_sync.py`：统一封装 `x_fin` 的数据刷新流程，避免训练/推理/实盘重复代码。
- `trainer.py`：PPO collect/update 主循环 + `torch.compile` 开关处理 + 定期验证。
- `validator.py`：`infer_codes` 全集回测与收益汇总。
- `train.py`：训练入口（仅读配置并调度）。
- `main.py`：实盘交易入口（按 `trade.bar_interval_minutes` 节奏，管理持仓与下单）。
- `trade_live.py`：实盘交易主逻辑（仓位管理、调仓、下单执行）。
- `main_infer.py`：离线推理入口（空仓辅助分析，可打印排序）。
- `value_stock_screener/app.py`：价值选股主流程（独立模块）。
- `value_stock_screener/data_store.py`：筛股模块数据缓存层。
- 数据拉取仍使用 `x_fin/x_ak.py` 与 `x_fin/x_bs.py`，不改数据来源。

---

## 9. 实施顺序（建议）

1. 先固定 `config.json` schema（尤其 `sampler` 与 `model.acceleration`）。
2. 重写采样器：严格按 `valid_start/valid_end` 计算可采样区间。
3. 实现 Tiny Transformer 提取器并接入 SB3 自定义 policy。
4. 接入 SDPA 与 `torch.compile`（提供一键开关）。
5. 打通训练-验证-最佳模型覆盖保存闭环。
6. 最后清理旧入口、旧参数和冗余 fallback。

---

## 10. 约束与风格

- 禁止分散配置：参数统一来自 `config.json`。
- 失败即报错，不做过度兜底。
- 单函数保持短小，单一职责。
- 日志聚焦关键信息，不写冗长分支。
