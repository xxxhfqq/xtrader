# 项目说明与配置详解

本文档对应当前代码状态，重点说明：

- 两个 `window_size` 的区别与关系
- 当前收益定义（是否对数收益）
- `main.py` 现在在做什么
- `easytrader` 与自动下单现状
- `config.json` 全参数说明

---

## 1. 当前项目在做什么

当前项目分为三条主链路：

1. 训练链路：`train.py` -> `trainer.py`
2. 离线推理链路：`main_infer.py`
3. 实盘交易链路：`main.py` -> `trade_live.py`

训练时会：

- 读取 `config.json`
- 用 `x_fin/x_bs.py` + `x_fin/x_ak.py` 更新数据
- 加载多股票数据并计算指标
- 在多股票随机采样环境里训练 PPO（自定义 Tiny Transformer 特征提取器）
- 按固定 rollout 周期做验证，并仅覆盖保存最佳模型

离线推理时会：

- 读取 `config.json`
- 可选更新推理数据
- 加载最佳模型
- 对目标股票计算动作信号并排序打印（TopK）

实盘时会：

- 连接 `easytrader` 客户端
- 按 `trade.bar_interval_minutes` 配置节奏更新数据、推理并执行委托
- 复用当前模型推理内核（多标共享模型）

---

## 2. 两个 window_size 的区别

你指出的是：

- `sampler.infer_window_bars`
- `model.features_extractor.window_size`

语义上：

- `sampler.infer_window_bars`
  - 属于采样/环境层参数
  - 表示每次决策时，环境要提供多少根 K 线窗口（obs 序列长度）
- `model.features_extractor.window_size`
  - 属于模型结构参数
  - 表示 Tiny Transformer 接收的序列长度

当前实现中，这两者**必须相等**，不是可独立配置：

- `config.py` 的 `validate_config()` 里做了强校验，不相等会直接报错。

为什么要强制相等：

- 环境输出的序列长度和特征提取器输入长度必须一致，否则维度不匹配。

结论：

- 现在你把其中一个改大/改小，另一个也要同步改。

---

## 3. 当前是否使用对数收益率

当前默认使用**对数收益率**（可配置）：

- 在 `env_stock.py` 里，reward 定义为：
  - 当 `env.reward_type = "log_return"` 时：`reward = log(now_equity / prev_equity)`
  - 当 `env.reward_type = "equity_delta"` 时：`reward = now_equity - prev_equity`

时序口径是：

- 在 `bar_t` 收盘后决策
- 在 `bar_{t+1}` 开盘执行买卖
- 然后用当前步权益变化作为奖励

为避免除零，环境会使用 `env.reward_epsilon` 做最小值保护。

---

## 4. 现在的 main.py 在做什么

当前 `main.py` 已恢复为**实盘主入口**，直接执行实盘交易循环并连接 `easytrader`。

目前职责拆分为：

- `main.py`：实盘交易入口（按 `trade.bar_interval_minutes` 节奏推理 + 下单）
- `main_infer.py`：空仓辅助推理与信号排序（不下单）

交易推理底层已经接到当前新框架模型（`trained_model/multi_symbol/best_model.zip`），且不再通过旧 `run2` 桥接。

---

## 5. easytrader 现状与下单问题

`easytrader` 执行层已恢复，并补回了 `x_fin/easytrader_patch.py` 验证码补丁。

说明：

- 交易主循环来自找回文件并接入新推理接口
- 推理动作输出 `[-1, 1]`，先线性映射为目标仓位 `[0, 1]`（`target=(signal+1)/2`）
- 实盘下单前还会经过阈值离散化与交易约束（最小交易额、整手、可用资金/可卖股数等），最终成交仓位不与原始动作严格等价
- 仍建议先在模拟/小仓位环境验证券商端字段与委托行为

---

## 6. value_stock_screener 说明

`value_stock_screener` 现已恢复为独立选股模块（非候选池占位脚本），核心文件为：

- `value_stock_screener/app.py`
  - 主板价值筛选主流程（数据更新、打分、行业分散、结果导出）
- `value_stock_screener/data_store.py`
  - 本地数据缓存与增量合并（`app_data/` 下各类 csv + meta）
- `value_stock_screener/__init__.py`
  - 包级入口（`run_screener()`），内部调用 `value_stock_screener.app.main`

当前行为：运行筛股后会在 `app_logs/` 输出 txt 报表与 json 代码列表（TopK）。

---

## 7. config.json 参数详解

以下为当前 `config.json` 字段的用途说明。

### 7.1 paths

- `paths.data_dir`
  - 数据目录，默认 `data`
  - 读取 `{code}.csv` 与 `{code}_infer.csv`
- `paths.model_dir`
  - 模型根目录，当前训练输出在 `trained_model/multi_symbol`
- `paths.log_dir`
  - 日志目录保留字段（当前核心日志写在 model 目录下）

### 7.2 通用运行参数

- `topK`
  - 在 `main.py` 中表示“从 `trade_codes` 候选池中最多管理多少只交易标的（槽位数）”
  - 在 `main_infer.py` 中仅用于额外展示前 K 个信号，不影响全量推理与排序
- `run_full_infer`
  - `true`：`main.py` 启动与跨日时，对全标的（`train_codes + infer_codes + trade_codes`）按“空仓状态”做一次推理并排序；`main_infer.py` 同样对该全量集合做推理与全量排序
  - `false`：`main.py` 跳过上述全标的快照；`main_infer.py` 仅对 `trade_codes`（若为空则回退 `infer_codes`）推理并排序

### 7.3 股票列表

- `train_codes`
  - 训练时可被随机采样的股票池
- `infer_codes`
  - 验证/推理股票池
- `trade_codes`
  - 交易关注股票池（结构：`code` + `name`）
  - `trade_live.py` 候选池默认直接读取该字段

### 7.4 features

- `features.indicators`
  - 技术指标列表，由 `stockstats` 计算
  - 与环境 OHLCV 特征拼接后作为模型输入

### 7.5 sampler

- `sampler.rollout_steps`
  - 每个 episode rollout 长度
  - 当前实现要求等于 `ppo.n_steps`
- `sampler.infer_window_bars`
  - 环境观测窗口长度（每次输入多少根 K 线）
- `sampler.indicator_warmup_bars`
  - 指标预热条数映射
  - 用于计算可采样起点，避免指标不完整
- `sampler.extra_warmup_bars`
  - 额外预热冗余
- `sampler.next_open_guard_bars`
  - 为“下一根开盘成交”预留的尾部保护条数
- `sampler.valid_start_formula`
  - 文档化公式字符串（说明用途）
- `sampler.valid_end_formula`
  - 文档化公式字符串（说明用途）

实际计算逻辑：

- `warmup = max(max(indicator_warmup_bars.values()), extra_warmup_bars)`
- `valid_start = (infer_window_bars - 1) + warmup`
- `valid_end = data_len - rollout_steps - next_open_guard_bars - 1`

补充说明（重点）：

- 上述 `valid_end` 公式用于**训练采样**。
- 单步推理（`main_infer.py` 与 `main.py` 实盘）使用最新可决策位置：
  - `infer_start_idx = data_len - next_open_guard_bars - 1`
- 这样不会被训练用 `rollout_steps`（如 4096）错误限制，避免 `_infer.csv` 行数较短时无法推理。

### 7.6 model

- `model.mode`
  - 当前为 `tiny_transformer`
- `model.features_extractor.name`
  - 特征提取器类名
- `model.features_extractor.architecture`
  - 当前支持 `decoder_only`
- `model.features_extractor.window_size`
  - Transformer 输入序列长度
  - 当前需与 `sampler.infer_window_bars` 一致
- `model.features_extractor.d_model`
  - 隐层维度
- `model.features_extractor.n_heads`
  - 多头数
- `model.features_extractor.n_layers`
  - 解码层数
- `model.features_extractor.ffn_dim`
  - 前馈层宽度
- `model.features_extractor.dropout`
  - Dropout 概率
- `model.features_extractor.pooling`
  - 序列池化方式，`last` 或 `mean`

- `model.policy_head.pi`
  - 策略头 MLP 结构
- `model.policy_head.vf`
  - 价值头 MLP 结构

- `model.acceleration.use_sdpa`
  - 是否启用 `scaled_dot_product_attention`
- `model.acceleration.use_torch_compile`
  - 是否启用 `torch.compile`
- `model.acceleration.torch_compile_mode`
  - compile 模式
- `model.acceleration.torch_compile_fullgraph`
  - compile fullgraph 开关
- `model.acceleration.torch_compile_dynamic`
  - compile dynamic 开关

### 7.7 ppo

- `ppo.learning_rate` 学习率
- `ppo.n_steps` rollout 长度（与 `sampler.rollout_steps` 一致）
- `ppo.batch_size` mini-batch 大小
- `ppo.n_epochs` 每轮更新 epoch 数
- `ppo.gamma` 折扣因子
- `ppo.gae_lambda` GAE 参数
- `ppo.clip_range` PPO clip
- `ppo.ent_coef` 熵正则

### 7.8 train

- `train.seed` 随机种子
- `train.total_timesteps` 总训练步数
- `train.validate_every_rollouts` 每多少个 rollout 做一次验证
- `train.device` 训练与推理默认设备
- `train.update_data_before_train`
  - 训练前是否先更新行情数据
- `train.update_data_before_infer`
  - 推理前是否先更新行情数据

### 7.9 env

- `env.initial_cash` 初始现金
- `env.buy_fee_rate` 买入费率
- `env.sell_fee_rate` 卖出费率
- `env.lot_size` 最小交易单位（A 股一般 100）
- `env.reward_type`
  - 奖励类型：`log_return` 或 `equity_delta`
- `env.reward_epsilon`
  - 对数收益计算时的最小保护值（必须 > 0）

### 7.10 trade

- `trade.min_fee`
  - 交易最低手续费（用于实盘下单股数估算）
- `trade.open_full_threshold`
  - 推理目标仓位高于该阈值时按满仓处理
- `trade.flat_threshold`
  - 推理目标仓位低于该阈值时按清仓处理
- `trade.open_candidate_threshold`
  - 新候选标加入持仓管理的最低目标仓位阈值
- `trade.rebalance_diff_threshold`
  - 实际仓位与目标仓位差值小于该阈值时跳过调仓
- `trade.min_trade_value`
  - 单次调仓最小成交金额阈值（买卖都生效）
- `trade.bar_interval_minutes`
  - 实盘主循环的 bar 触发粒度（默认 5 分钟）
- `trade.order_settle_initial_wait_seconds`
  - 每轮委托后，进入成交检测前的初始等待秒数
- `trade.order_settle_max_wait_seconds`
  - 每轮委托后，最多等待成交回报的总秒数
- `trade.in_bar_poll_sleep_seconds`
  - 交易时段内、尚未到下一 bar 时的轮询休眠秒数
- `trade.out_of_trading_sleep_seconds`
  - 非交易时段轮询休眠秒数
- `trade.xiadan_path`
  - 同花顺下单客户端路径（可被 `main_config.json.xiadan_path` 运行时覆盖）
- `trade.trading_sessions`
  - 交易时段配置，格式为 `[[start, end], ...]`，时间格式 `HH:MM[:SS]`

### 7.11 save

- `save.best_model_name`
  - 最佳模型文件名（无后缀）
- `save.best_log_file`
  - 验证日志文件名（jsonl）

---

## 8. 常见改法示例

### 8.1 把窗口从 240 改成 480

必须同时改：

- `sampler.infer_window_bars = 480`
- `model.features_extractor.window_size = 480`

### 8.2 让验证更频繁

- 调小 `train.validate_every_rollouts`，例如从 `1000` 改到 `200`

### 8.3 想降低显存占用

- 降低 `d_model` 或 `n_layers`
- 关闭 `use_torch_compile`（部分环境会额外吃显存）

---

## 9. 你当前最关心的结论（简版）

- 两个 `window_size` 语义不同，但当前实现要求相等。
- 当前默认用对数收益率（可切到绝对权益增量）。
- 当前 `main.py` 为实盘入口；`main_infer.py` 为离线推理入口。
- `main.py` 的 `topK` 是“从 `trade_codes` 候选池中可管理的持仓槽位数”，不是“按分数选前 K 名”。
- 单步推理已改为“最新 bar 决策索引”，不再受训练 `rollout_steps` 影响。
- `value_stock_screener` 已恢复为独立筛股模块（`app.py + data_store.py`），不再是占位脚本。

---

## 10. data CSV 字段对齐核对（关键）

先给结论：**当前 `data/` 下 CSV 字段与代码期望是对齐的**。

代码侧对字段的强依赖：

- `features.py` 中 `RAW_COLUMNS` 明确要求：
  - `["date", "code", "time", "open", "high", "low", "close", "volume"]`
- `x_fin/x_bs.py` 的 `get_data_from_bs()` 默认拉取字段：
  - `date,time,code,open,high,low,close,volume`
- `x_fin/x_ak.py` 的 `update_infer_data_from_ak()` 在落盘前也会重排为同一字段顺序：
  - `["date", "code", "time", "open", "high", "low", "close", "volume"]`

数据侧核对结果：

- 当前 `data/` 目录共 `37` 个 `_infer.csv`。
- 表头均为：
  - `date,code,time,open,high,low,close,volume`
- 与 `features.load_symbol_frame()` 的必要列检查完全一致，不会因字段缺失触发 `KeyError`。

建议持续检查点（防回归）：

- 新增数据源时，确保列名仍为英文标准字段（不要回退到中文列名如“时间/开盘/收盘”）。
- 若未来改动字段，请同步更新：
  - `features.py` 的 `RAW_COLUMNS`
  - `x_fin/x_bs.py` 的拉取字段
  - `x_fin/x_ak.py` 的重命名与列重排逻辑

---

## 11. 项目结构与模块级职责

### 11.1 结构总览（按目录）

```text
fin/
├─ main.py
├─ train.py
├─ main_infer.py
├─ trade_live.py
├─ trainer.py
├─ validator.py
├─ env_stock.py
├─ sampler.py
├─ features.py
├─ policy.py
├─ extractor_tiny_transformer.py
├─ data_sync.py
├─ config.py
├─ config.json
├─ requirements.txt
├─ prompt.md
├─ PROJECT_GUIDE.md
├─ .gitignore
├─ data/
├─ app_logs/
├─ deleted_files/
├─ x_fin/
└─ value_stock_screener/
```

### 11.2 根目录文件职责

- `main.py`
  - 实盘入口；直接调用 `trade_live.main()`。
- `train.py`
  - 训练入口；调用 `trainer.train_from_config("config.json")`。
- `main_infer.py`
  - 离线/实时推理入口；加载最佳模型并输出每个标的信号与目标仓位。
- `trade_live.py`
  - 实盘主循环：连接券商、按 bar 节奏刷新数据、推理、下单与调仓。
- `trainer.py`
  - 训练主逻辑：数据加载、环境构建、PPO 训练、验证回调、best/last 模型保存。
- `validator.py`
  - 训练中验证器：在 `infer_codes` 上回测式评估并汇总 PnL。
- `env_stock.py`
  - Gym 环境：账户状态、T+1 约束、成交撮合与 reward 计算。
- `sampler.py`
  - 采样器与区间计算：随机多标采样、固定 episode 采样、warmup/valid range 逻辑。
- `features.py`
  - 行情 CSV 加载与指标工程：字段校验、时间解析、stockstats 指标生成。
- `policy.py`
  - 策略构建辅助：市场特征列定义、SB3 `policy_kwargs` 组装。
- `extractor_tiny_transformer.py`
  - 自定义 SB3 特征提取器：轻量 `decoder-only` Transformer 实现。
- `data_sync.py`
  - 统一数据刷新编排：串联 `x_bs`（历史）与 `x_ak`（infer/比例）更新。
- `config.py`
  - 配置加载与强校验：默认值填充、字段类型与约束检查。
- `config.json`
  - 运行配置源：股票池、采样、模型、训练、交易与保存参数。
- `requirements.txt`
  - 依赖清单：交易、数据、RL、指标与系统依赖。
- `prompt.md`
  - 历史/设计草案文档：重构目标、采样规则、模型路线说明。
- `PROJECT_GUIDE.md`
  - 当前说明文档：参数详解、运行链路、关键结论与项目结构说明。
- `.gitignore`
  - Git 忽略规则（如 `*.csv`、`*.zip`、`__pycache__/` 等）。

### 11.3 目录职责

- `data/`
  - 行情数据目录；当前主要为 `{code}_infer.csv`（推理使用）。
- `app_logs/`
  - 运行日志与筛股输出日志目录（如 `select_*.txt`）。
- `deleted_files/`
  - 预留目录（当前为空）。
- `x_fin/`
  - 行情与交易适配层（BaoStock/AkShare/easytrader 补丁）。
- `value_stock_screener/`
  - 独立价值选股模块（数据更新、打分筛选、结果导出）。

### 11.4 子模块文件职责

- `x_fin/__init__.py`
  - 对外导出数据更新与行情读取函数。
- `x_fin/x_bs.py`
  - BaoStock 5 分钟历史数据拉取、增量更新、数据完整性验证与 infer 文件生成。
- `x_fin/x_ak.py`
  - AkShare 行情补齐、买卖盘/现价获取、复权比例估算与 infer 合并更新。
- `x_fin/easytrader_patch.py`
  - `easytrader` 验证码与表格读取猴子补丁，提升自动化稳定性。
- `value_stock_screener/__init__.py`
  - 选股模块包级入口（`run_screener`）。
- `value_stock_screener/app.py`
  - 价值选股主流程：股票池获取、财务/估值数据更新、评分与行业分散、结果落盘。
- `value_stock_screener/data_store.py`
  - 选股数据存储层：`app_data/` 内各表读写、增量合并、去重与元数据管理。
