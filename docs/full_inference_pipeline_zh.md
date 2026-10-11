# 完整 Bar 推理流水线说明

本文是当前选中生产方案的主说明文档，覆盖从 canonical indicator 到因子、模型信号、
组合求解和 target position 的完整流程，以及离线重放、在线 SHM、异步 Parquet、恢复、
资源约束和一致性验证。

第一次在新机器部署时，请先按 `docs/reproduce_selected_pipeline_zh.md` 固定三个仓库版本、
使用同一 Python 解释器编译 `teletubbies-alpha` pybind `_core.so`，并完成最小 parity 验收；
不要直接复用其他机器编译出的 SO。

当前对应配置：

- 离线全链路：`configs/offline_selected_pipeline_5m.yaml`
- 在线全链路：`configs/online_selected_pipeline_5m.yaml`
- 因子清单：`teletubbies-alpha/results/factors/alpha_inventory_1111_strict.yaml`

这些路径记录的是截至 `2026-01-31` 的已验证数据快照。连接新的 live bar 前，必须更新
canonical parquet、metadata 和 `runtime.prewarm.end_day`，使 prewarm 最后一根恰好是首根
live bar 的前一根。不能把当前快照配置原样连接到更晚日期。

## 1. 生产契约

当前方案处理 5 分钟 bar：

```text
canonical indicator
    -> signal runner: 1111 个流式因子 + valid-only CS z-score
    -> model runner: LGBM ret1h + EMA MLP ret4h + EMA MLP ret8h
    -> portfolio runner: 约束多空组合求解
    -> target_weight / target_qty
```

在线数据面：

```text
bar_indicator.shm
    -> bar_signal.shm
    -> bar_model.shm
    -> bar_portfolio.shm
    -> target_position.shm
```

在线每个 stage 都先提交 SHM。Parquet 只作为异步镜像，不参与下游依赖，也不允许
磁盘 I/O 阻塞主推理链路。

当前选中生产参数：

| 项目 | 当前值 |
| --- | --- |
| bar 周期 | 5 分钟 |
| 每天 bar 数 | 288 |
| symbols | 617，必须与模型 manifest 顺序一致 |
| alpha 数 | 1111 |
| alpha 输出 | valid-only CS z-score，clip 到 `[-20, 20]` |
| 模型 | LGBM ret1h、EMA MLP ret4h、EMA MLP ret8h |
| 模型融合 | 三个分量分别 CS z-score，等权平均，再做一次 CS z-score |
| book size | 5,000,000 USD |
| 求解频率 | 每个 5 分钟 bar |
| 规划成交率 | 过去成交额估计的 2% |
| 单票权重上限 | 1% |
| 净敞口上限 | 10% book |
| 单 bar 换手上限 | 100% book |
| candidate pool | 多空各 75，加已有持仓 |
| 4 小时风险预算 | 350 bps |
| CPU 上限 | 16 个逻辑 CPU |
| RSS 上限 | 62 GiB |

## 2. 输入数据契约

### 2.1 离线输入

当前 canonical 数据：

```text
/data/indicator/prod_v1_bar_5m/concat/2024-08-01_2026-01-31.parquet
/data/indicator/prod_v1_bar_5m/concat/2024-08-01_2026-01-31.metadata.json
```

metadata 是 schema、日期、symbol 顺序和字段顺序的权威来源。运行时要求：

- 每天恰好 288 根 bar，`ti=0..287`。
- `timestamp_us` 是真实 bar 时间，连续 5 分钟应相差 `300000000` 微秒。
- indicator 数值输入是 `float32`。
- symbol 顺序不能在 stage 之间重排。
- schema hash、字段顺序、symbol 顺序不一致时 fail-fast。

原始 `/data/crypto` 保持只读。新的推理结果放在 `/data/features`。

### 2.2 在线输入

Rust bar 聚合器写入 `bar_indicator.shm`。Python 读取最新 committed frame，检查：

- SHM 协议版本。
- schema hash。
- sequence 单调递增。
- sequence gap。
- heartbeat freshness。
- `timestamp_us` 连续性。

`position.shm` 由交易或账户同步侧写入。portfolio runner 使用实际持仓，而不是假设上一次
target 已经完全成交。

## 3. Signal Runner

Signal stage 顺序执行三个 module：

1. `ProdV15mBarFeatureSet`
2. `KDataBarFeatureSet`
3. `AlphaExpressionSignal`

`KDataBarFeatureSet` 提供 portfolio 和模型所需的：

```text
open, high, low, close, vwap, valid, trade_amt
```

`AlphaExpressionSignal` 从严格筛选后的 1111 因子 YAML 加载表达式。当前关键参数：

```yaml
execution: streaming
accumulation: fast
num_threads: 16
output_normalization: cszscore
valid_column: valid
cszscore_min_count: 2
cszscore_eps: 1.0e-12
output_clip_abs: 20.0
lookback: 864
num_tis: 288
require_streamable: true
```

每根 bar 的处理逻辑：

1. 更新流式算子内部状态。
2. 计算所有原始因子。
3. 仅对 `valid != 0` 且数值有限的 symbols 做截面 z-score。
4. 将最终因子裁剪到 `[-20, 20]`。
5. invalid 点输出 NaN，不进入模型和后续截面统计。

当前默认搜索表达式不含截面算子，截面变换只发生在最终统一的 CS z-score 层。

### 3.1 状态和回看

内存 store 最多保留 864 根 bar，即 3 天。算子内部状态也以流式方式维护，不保存完整历史
`(T,N,F)`。

当前选中 fast accumulator 因子清单要求从 `2024-11-28` state anchor 重放。不能在任意
日期只读取最近 3 天后直接声称 endpoint 等价。在线首次启动或异常重启后，当前正确做法仍是
从 anchor 重放 canonical indicator，重建因子状态。

## 4. Model Runner

模型 module 为 `LgbmMlpEnsembleModel`。输入为 1111 个因子和 `valid`，模型组件是：

1. LightGBM `ret1h` 模型。
2. EMA MLP `ret4h` ONNX 模型。
3. EMA MLP `ret8h` ONNX 模型。

当前模型文件：

```text
/data/features/alpha/models/lgbm_ret1h_csdemean_l512_lr0p05_bin63_r50_train20241130_20250630_eval2025q3_v1/model.txt
/data/features/alpha/models/mlp_raw_all_targets_csdemean_bs16_lr5e-4_epoch1_train20241130_20250630_oos20250701_20260131_v1/models/ret_4h/model_ema.onnx
/data/features/alpha/models/mlp_raw_all_targets_csdemean_bs16_lr5e-4_epoch1_train20241130_20250630_oos20250701_20260131_v1/models/ret_8h/model_ema.onnx
```

LGBM 是 50 rounds、512 leaves、learning rate 0.05、feature fraction 0.9、
bagging fraction 0.9、max bin 63 的 ret1h 模型。

两个 MLP 的部署结构相同：

```text
BatchNorm1d(1111)
Dropout(0.3)
Linear(1111, 256, bias=False)
BatchNorm1d(256) -> GELU -> Dropout(0.3)
Linear(256, 128, bias=False)
BatchNorm1d(128) -> GELU -> Dropout(0.3)
Linear(128, 1, bias=False)
```

MLP 使用 CCC loss、batch size 16 个截面、learning rate `5e-4`、1 epoch、EMA 0.999。
部署 ONNX 使用 EMA parameters；BatchNorm buffers 在 EMA 更新时直接 copy，不做指数平滑。

启动时执行以下校验：

- 因子 YAML、LGBM 和两个 ONNX 文件的 SHA256。
- LGBM feature name 与因子清单完全一致。
- ONNX 输入名为 `features`，输出名为 `prediction`。
- ONNX feature 数为 1111。
- symbol manifest 与 runtime symbol 顺序完全一致。
- 请求的 ONNX Runtime provider 可用。

有效模型点定义为：

```text
valid != 0
and valid is finite
and all 1111 input features are finite
```

融合公式：

```text
z_lgbm = CSZ(lgbm_ret1h_raw | effective_valid)
z_mlp4 = CSZ(mlp_ret4h_raw | effective_valid)
z_mlp8 = CSZ(mlp_ret8h_raw | effective_valid)

ensemble_mean = (z_lgbm + z_mlp4 + z_mlp8) / 3
meta_signal = CSZ(ensemble_mean | effective_valid)
```

权重可以配置，当前是 `[1, 1, 1]` 归一化后的等权。

Model stage 输出：

```text
lgbm_ret1h_raw
mlp_ret4h_raw
mlp_ret8h_raw
lgbm_ret1h_cszscore
mlp_ret4h_cszscore
mlp_ret8h_cszscore
ensemble_mean
meta_signal
output_valid
```

portfolio 使用最终 `meta_signal`。

## 5. Portfolio Runner

portfolio module 为 `ConstrainedLongShortPortfolio`。每根 5 分钟 bar 都重新求一次 target。

### 5.1 输入

- `model.meta_signal`
- `signal.valid`
- `signal.vwap`
- `signal.trade_amt`
- `position.current_weight`

`valid == 0` 的票不会新开仓。已有持仓会继续进入候选 universe，求解器只能根据容量约束
维持或减仓，不能因为缺失信号扩大风险。

### 5.2 容量估计

规划容量使用过去 48 根 bar 的正成交额中位数：

```text
prior_capacity_weight
    = median(previous 48 bars trade_amt) * 2% / book_size_usd
```

单票最大持仓还受过去 48 根 bar 总成交额限制：

```text
max_position_weight
    = sum(previous 48 bars trade_amt) * 5% / book_size_usd
```

这是先验约束，只使用当前 bar 之前已经观察到的数据，不读取下一根 bar。

### 5.3 风险估计

使用过去 288 根 5 分钟 VWAP return 估计波动率，再按 48 根 bar 缩放为 4 小时风险。
只有连续时间戳、前后都 valid 且价格为正的 return 才进入统计。有效观测少于 96 时，该票
风险估计记为缺失，并按风险预算处理。

当前风险预算为 350 bps，风险 slack penalty 为 100。

### 5.4 优化目标

求解器使用 SciPy HiGHS 线性规划。等价目标是最小化：

```text
- signal * target_weight
+ turnover_cost * abs(target_weight - current_weight)
+ net_penalty * abs(sum(target_weight))
+ risk_slack_penalty * risk_slack
```

当前：

```text
turnover_cost = 0.65
net_penalty = 5.0
risk_slack_penalty = 100.0
```

这里的 `turnover_cost=0.65` 是归一化信号空间中的优化惩罚，不是直接等于 65 bps 或实盘
手续费。

主要约束：

- gross weight 不超过 1.0。
- 绝对 net weight 不超过 0.10。
- 单根 bar 计划换手不超过 1.0。
- 单票交易变化不超过先验容量。
- 单票绝对持仓不超过 1%。
- 单票持仓同时受过去 4 小时成交额容量限制。
- 高波动股票的有效单票上限按风险比缩小。
- 组合 4 小时风险 proxy 不超过 `risk_budget * (1 + risk_slack)`。

候选 universe 是 `meta_signal` 最强的多空各 75 只，加所有已有持仓。加入已有持仓是为了
保证已有风险总能进入求解器并有机会减仓。

### 5.5 求解失败

依次尝试：

```text
highs -> highs-ds -> highs-ipm
```

每次成功后还会检查约束可行性和数值有限性。全部失败时不会保持任意旧 target，也不会让
进程因为未处理异常直接退出，而是进入确定性的 reduce-only fallback：

- 优先配对减少多头和空头。
- 再减少导致净敞口的一侧。
- 不新开仓，不扩大已有绝对持仓。
- 仍受成交容量和换手限制。

### 5.6 输出

```text
target_weight
target_notional
target_qty
current_weight
rebalance_weight
ready
solver_success
gross_weight
net_weight
planned_turnover_weight
risk_proxy_4h
risk_slack
```

`target_qty = target_weight * book_size_usd / vwap`。VWAP 缺失或非正时，target quantity 为 0。

`target_position.shm` 对交易侧暴露当前需要的：

```text
target_weight, target_qty
```

资金费率、成交滑点、PnL、Sharpe 和 Calmar 属于回测评估层，不在在线 target-weight
推理模块里计算。

## 6. 离线全链路重放

离线入口：

```bash
cd /home/kemove/code/teletubbies/teletubbies-py
PYTHONPATH=../teletubbies-alpha/build-cuda/python:src \
  /home/kemove/anaconda3/envs/rainbow/bin/python \
  -m teletubbies_py.tools.run_offline_streaming_pipeline \
  --config configs/offline_selected_pipeline_5m.yaml
```

离线 runner 仍按逐 bar 因果顺序运行 signal、model、portfolio，不把完整数据加载进内存。
canonical parquet 按 1 天 chunk 读取，rolling store 只保留有限历史。

默认输出：

```text
/data/features/alpha/inference/lgbm1h_mlp4h8h_book5m_target_only_v2/
    timeline.npy
    target_weight.npy
    manifest.json
```

NPY 是标准 `.npy`，由 streaming writer 写入，使用 `np.load(path, mmap_mode="r")` 读取。
`target_weight.npy` shape 为：

```text
(T, N, 1)
```

dtype 为 `float32`。`timeline.npy` 保存：

```text
YYYYMMDD, ti, di, timestamp_us
```

因子 `signal.npy` 默认不写，避免 1111 因子造成额外大文件。需要时显式传：

```bash
--output-dir /data/features/alpha/inference/<run_name> --write-signal
```

### 6.1 离线 target feedback

该 runner 只负责推理 target，不模拟成交、fill rate、强平或 PnL。bar `t` 产生的
`target_weight` 在 bar `t+1` 作为求解器的参考 `current_weight`：

```text
position_feedback: previous_target
```

这是一个因果、确定性的 target-only 反馈，用于在没有实盘 position feed 的离线运行中计算
换手和下一目标。它不读取下一根 bar 的真实成交额，也不声称 target 已真实成交。

`previous_target` 适合独立生成 target，但不能保证复现存在部分成交时的历史 target。需要验证
回测一致性时，配置 `position_feedback.mode: npy`，逐 bar 重放当时已经可见的
`current_weight`。该模式只读取仓位输入，不在 runner 内计算 fill 或 PnL；实盘对应输入仍是
`position.shm`。

后验成交率、实际 fill、invalid 强平、交易成本、funding 和 PnL 必须在独立回测 evaluator
中完成。`offline_streaming_pipeline` 如果发现 `execution_replay` 配置会直接拒绝启动，避免把
回测simulation误混入推理产物。

## 7. 在线全链路

当前选中生产路径使用单进程 `OnlineBarRuntime`，在一个 Python 进程中严格串行执行：

```text
read bar_indicator frame X
-> signal(X)
-> commit bar_signal SHM X
-> model(X)
-> commit bar_model SHM X
-> portfolio(X)
-> commit bar_portfolio and target_position SHM X
-> enqueue all enabled Parquet mirrors
```

stage 之间在同一进程中通过 `BarContext` 和 `LiveOutputStore` 直接传值。SHM 同时作为外部
可观测输出和未来拆分进程的协议边界。

### 7.1 初始化 SHM

```bash
cd /home/kemove/code/teletubbies/teletubbies-py
PYTHONPATH=../teletubbies-alpha/build-cuda/python:src \
  /home/kemove/anaconda3/envs/rainbow/bin/python \
  -m teletubbies_py.tools.init_shm \
  --pipeline-config configs/online_selected_pipeline_5m.yaml \
  --metadata /data/indicator/prod_v1_bar_5m/concat/2024-08-01_2026-01-31.metadata.json \
  --base-dir /dev/shm/teletubbies_selected_5m \
  --ring-size 16 \
  --dtype float32
```

SHM 协议当前为 v2。升级协议或修改 symbols/columns 后必须重新初始化，不能复用旧文件。

### 7.2 启动 runtime

```bash
systemd-run --user --scope \
  -p MemoryHigh=58G \
  -p MemoryMax=62G \
  -p MemorySwapMax=0 \
  taskset -c 0-15 env \
  PYTHONPATH=../teletubbies-alpha/build-cuda/python:src \
  /home/kemove/anaconda3/envs/rainbow/bin/python \
  -m teletubbies_py.tools.run_online_runtime \
  --config configs/online_selected_pipeline_5m.yaml
```

启动顺序：

1. 校验模型、因子、metadata、manifest 和 checksum。
2. 初始化或重新初始化全部 SHM。
3. 启动 Python runtime，完成 signal state prewarm。
4. 确认 prewarm 完成且资源未超限。
5. 启动 live bar writer。
6. live 首根 bar 必须恰好比 prewarm 最后一根晚 5 分钟。
7. 确认 position feed freshness 后允许交易侧消费 target position。

### 7.3 分进程 worker

项目也提供 signal、model、portfolio 三个独立 worker 和 `SequenceBarrier`。拆分后拓扑是：

```text
bar_indicator -> signal worker -> bar_signal
bar_signal -> model worker -> bar_model
bar_model + position -> portfolio worker -> target_position
```

每个 worker 在 SHM commit 后先通知下游，再把 Parquet 镜像放入后台队列。

当前选中 1111 因子生产部署仍应使用单进程 `OnlineBarRuntime`，因为它实现了完整的 factor
state prewarm。独立 `SignalStageWorker` 尚未实现同等的历史预热，不能直接作为当前因子库的
生产替代。

## 8. SHM 协议和同步

每个 SHM 是 mmap 文件：

```text
4096-byte header
+ ring_size * num_symbols * num_columns * sizeof(dtype)
```

当前使用 `float32`、ring size 16。选中 schema 总 SHM 约 55 MiB。

writer 使用 seqlock：

1. header epoch 写为奇数。
2. 写目标 ring slot。
3. header epoch 写为偶数并提交 sequence。
4. 通知下游 eventfd，或由下游 polling 发现新 sequence。

reader 只有在两次 header 读取完全一致且 epoch 为偶数时才接受 frame。

关键字段：

- `sequence`：当前 stage 帧序号。
- `source_sequence`：对应的源 bar 序号。
- `heartbeat_ns`：进程存活和 freshness，不用于收益窗口。
- `timestamp_us`：真实 bar 时间，用于窗口连续性和风险计算。
- `di/ti`：交易日和日内 bar 索引。

任一 worker 看到上游 sequence 已经超过自己需要的 source sequence 时会 fail-fast，不会
静默跳帧后继续生成错位结果。

## 9. 异步 Parquet 镜像

online 配置为 signal、model、portfolio 都启用 Parquet mirror：

```yaml
output:
  parquet_mirror:
    root_dir: /data/features/alpha/online_selected_pipeline_5m
    queue_size: 64
    batch_bars: 12
    compression: zstd
```

输出目录：

```text
/data/features/alpha/online_selected_pipeline_5m/
    signal/trading_day=YYYY-MM-DD/part-<first>-<last>.parquet
    model/trading_day=YYYY-MM-DD/part-<first>-<last>.parquet
    portfolio/trading_day=YYYY-MM-DD/part-<first>-<last>.parquet
```

每行对应一个 `(sequence, symbol)`，固定元数据列为：

```text
sequence, source_sequence, di, ti, timestamp_us, symbol
```

其余列是该 stage 的输出列。

### 9.1 主链路保证

主线程只执行：

1. 把当前 stage frame 转为独立 `float32` 快照。
2. 使用 `put_nowait` 放入有界队列。

后台线程负责 Arrow 转换、Zstd 压缩和文件系统 I/O。文件先写
`.parquet.tmp-<pid>`，完成后在同一文件系统内 `os.replace` 为最终文件。

队列满、快照分配失败或磁盘写失败时：

- SHM 和 target position 不回滚。
- 在线推理不等待磁盘。
- 记录 `dropped_frames`、`write_errors` 和 `last_error`。
- 输出 ERROR 日志。
- 缺失区间后续从 canonical indicator 重放补齐。

正常退出会 drain 队列，并写出未达到 `batch_bars` 的最后一个批次。

### 9.2 持久性边界

当前 `batch_bars=12`。异常断电或强制 kill 时，最多可能丢失尚未封装成最终 part 的 11 根
bar。若需要每根 bar 尽快持久化，可以设为 1，但会显著增加小文件数量。

Parquet 镜像保存的是 stage 结果，不保存 TS 算子内部 deque、sum、sumsq 等状态。因此：

- 它可以做审计、结果恢复和下游重放。
- 它不是完整 factor state checkpoint。
- 当前进程重启仍需 canonical prewarm。

## 10. 资源和延迟

当前资源保护：

- CPU affinity 限制为 16 个逻辑 CPU。
- factor、LightGBM、ONNX Runtime 单模块线程数最大 16。
- `MemoryHigh=58G`、`MemoryMax=62G`、禁用 swap。
- offline 每天检查当前 RSS，超过配置立即失败。
- online prewarm 记录 peak RSS，超过 62 GiB 立即失败。

以下资源和延迟数字是 2026-09-18 在当前机器和选中配置上的验证快照，不应跨硬件直接外推。

已验证的资源数据：

- 864 bar online prewarm peak RSS：10.482 GiB。
- 4 天完整 offline chain peak RSS：12.12 GiB。
- 全历史运行中的典型进程 RSS 约 13.6 GiB；cgroup file cache 会让 memory.current 更高。

已测吞吐：

- 完整历史链路约 228.8 ms/bar，约 4.37 bar/s。
- 该值包含 Parquet 输入、1111 因子、三个模型、求解、NPY 输出和日志。
- 5 分钟实盘每 300 秒只有一根新 bar，平均延迟余量充足。
- `617 x 1111` Parquet mirror 快照入队约 1.05 ms。
- 同尺寸单帧后台单核 Zstd 写盘约 90 ms，不阻塞推理线程。

求解器不是当前端到端主瓶颈。单个 portfolio 的时间序列依赖上一根实际持仓，不能并行求解
不同 bar。HiGHS 对当前小规模 LP 强行增加线程通常没有收益。优化优先级应放在因子表达式、
共享子表达式和 Python 数据提取路径。

## 11. 一致性和验收

### 11.1 因子层

要求：

- endpoint consistency：窗口外历史改变不影响最终值。
- invalid isolation：`valid == 0` 的点不影响 valid 点。
- batch/streaming parity：误差小于 `1e-4`。
- 最终输出必须执行 valid-only CS z-score。
- 输出 clip 必须为 `[-20, 20]`。

当前 1111 清单是严格筛选后的生产清单。失败因子已从生产 inventory 过滤。

### 11.2 模型层

当前验证：

- 32 个 OOS 截面 native inference parity 最大误差 `3.248e-6`。
- mask mismatch 为 0。
- 完整 anchor offline replay 与直接模型推理最大误差 0。
- online prewarm 后第一根 live bar 与 offline replay 最大误差 0。

### 11.3 Portfolio 层

当前验证：

- planned solver 与选中回测参考在绝对误差 `1e-12` 内一致。
- backtest 专用的 posterior execution solver 与参考在绝对误差 `1e-12` 内一致；
  它不进入离线或在线 target runner。
- 首个 online solve 成功，gross weight 0.137353，net 约为 0，risk slack 为 0。
- target weight 和 target quantity 对 617 个 symbols 都是有限值。

### 11.4 SHM 和持久化

必须覆盖：

- SHM writer/reader roundtrip。
- ring wrap。
- sequence regression 和 gap。
- schema、symbol、column mismatch。
- stale heartbeat。
- source sequence barrier。
- target position 映射。
- Parquet 原子分片、partial batch flush 和错误统计。

当前异步 Parquet 相关定向测试加核心 SHM/model/portfolio 测试为 24 passed。完整选中链路验证
快照见 `docs/reports/selected_full_inference_pipeline_2026-09-18.md`。

### 11.5 2025Q3 target-only 回放

使用已验证模型信号和因果 `current_weight` position feed，重放 `2025-07-01..2025-09-30`：

- 26,496 bars，617 symbols。
- target shape `(26496, 617, 1)`。
- 与旧参考回测 planned target 逐元素完全一致，最大绝对误差 0，Pearson 1。
- 重放峰值 RSS 3.097 GiB。
- runner 内没有 posterior fill、强平、funding 或 PnL 计算。

产物目录：

```text
/data/features/alpha/inference/lgbm1h_mlp4h8h_target_only_2025q3_position_feed_v1/
    target_weight.npy
    timeline.npy
    q3_validation.json
    manifest.json
```

## 12. 监控要求

生产监控至少暴露：

- 最新 bar sequence/source sequence。
- 最新 `di/ti/timestamp_us`。
- 每 stage 延迟 mean、p95、p99、max。
- dropped sequence 数。
- position heartbeat age。
- solver success/fallback 次数。
- gross、net、planned turnover、risk proxy、risk slack。
- Parquet `enqueued/written/dropped/errors`。
- process RSS、cgroup memory、CPU 使用率。
- SHM schema hash 和模型 artifact checksum。

出现以下任一情况应停止产生新的风险，而不是继续追最新值：

- schema 或 artifact checksum 不一致。
- sequence 回退或无法恢复的 gap。
- 首根 live bar 与 prewarm 不连续。
- position feed stale。
- 非有限 target。
- 资源超过硬上限。

Parquet mirror 丢帧不应停止 target 生成，但必须告警并安排 replay 补数。

## 13. 当前限制和后续工作

1. 尚无通用的算子内部 state checkpoint。重启仍需从 anchor 重放。
2. split signal worker 尚未接入当前选中因子的完整 prewarm。
3. Parquet part 尚无自动日终 compaction，长期运行需要单独的合并和保留策略。
4. 当前 mirror queue 满时选择不阻塞主链路并丢弃持久化副本，需要监控闭环。
5. 在线 target-weight runtime 不包含订单生命周期、真实 fill、funding 和 PnL 记账。
6. 实盘交易侧必须持续把真实 position 写回 `position.shm`。

## 14. 代码位置

| 功能 | 文件 |
| --- | --- |
| stage runner 和 context | `src/teletubbies_py/core/runner.py`, `core/context.py` |
| offline streaming chain | `src/teletubbies_py/core/offline_streaming.py` |
| online runtime 和 workers | `src/teletubbies_py/core/online.py` |
| SHM protocol | `src/teletubbies_py/core/shm.py` |
| async Parquet mirror | `src/teletubbies_py/core/async_parquet.py` |
| 1111 factor module | `src/teletubbies_py/bar_signal/alpha.py` |
| selected model ensemble | `src/teletubbies_py/bar_model/ensemble.py` |
| portfolio runtime | `src/teletubbies_py/bar_portfolio/optimizer.py` |
| LP solver/fallback | `src/teletubbies_py/bar_portfolio/solver.py` |
| online production config | `configs/online_selected_pipeline_5m.yaml` |
| offline production config | `configs/offline_selected_pipeline_5m.yaml` |
| SHM details | `docs/shm_protocol_zh.md` |
| split workers | `docs/split_stage_workers_zh.md` |
| validation report | `docs/reports/selected_full_inference_pipeline_2026-09-18.md` |
