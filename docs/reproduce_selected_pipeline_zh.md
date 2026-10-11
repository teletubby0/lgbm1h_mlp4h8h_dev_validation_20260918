# 选中因子、模型与信号的逐步复现

本文面向首次接手项目的开发者，从三个干净 checkout 开始，复现当前 5 分钟生产方案：

```text
framework canonical indicator parquet
  -> teletubbies-py streaming signal runner
  -> teletubbies-alpha pybind _core.so 计算 1111 个因子
  -> LGBM ret1h + EMA MLP ret4h + EMA MLP ret8h
  -> 分量 CS-zscore -> 等权平均 -> 最终 CS-zscore
  -> meta_signal
```

组合求解属于下一阶段；本文先保证因子、三个模型分量和最终信号可复现。完整 portfolio
说明见 `docs/full_inference_pipeline_zh.md`。

## 1. 固定代码版本

交付包验证时使用以下 `main` 提交：

| 仓库 | GitHub | 提交 |
| --- | --- | --- |
| teletubbies-framework | `git@github.com:teletubby0/teletubbies-framework.git` | `64019fccb75823b47e90106ca104d7b1838ce1af` |
| teletubbies-alpha | `git@github.com:teletubby0/teletubbies-alpha.git` | `24e3e6c6774e02dec5b2e8f2206d4f774f70075d` |
| teletubbies-py | `git@github.com:teletubby0/teletubbies-py.git` | `c918a5fc0cb8d18283f7d87c6b1c780f651d25b5` |

为了精确复现，应 checkout 表中的提交。若改用更新的 `main`，必须重新运行测试并更新模型、
因子配置和 schema 的 SHA256，不能混用旧交付包的 hash。

```bash
export TT_ROOT=/path/to/teletubbies
mkdir -p "$TT_ROOT"
cd "$TT_ROOT"

git clone git@github.com:teletubby0/teletubbies-framework.git
git clone git@github.com:teletubby0/teletubbies-alpha.git
git clone git@github.com:teletubby0/teletubbies-py.git

git -C teletubbies-framework checkout 64019fccb75823b47e90106ca104d7b1838ce1af
git -C teletubbies-alpha checkout 24e3e6c6774e02dec5b2e8f2206d4f774f70075d
git -C teletubbies-py checkout c918a5fc0cb8d18283f7d87c6b1c780f651d25b5
```

## 2. Python 环境

要求 Python 3.11、CMake 3.20+、C++20 编译器。CUDA 构建还需要 CUDA Toolkit 和兼容
驱动。内部机器默认使用 `rainbow`：

```bash
export PYTHON=/home/kemove/anaconda3/envs/rainbow/bin/python
"$PYTHON" -V
"$PYTHON" -m pip install pybind11 numpy
"$PYTHON" -m pip install -e "$TT_ROOT/teletubbies-py[onnx,test]"
```

新机器可使用独立 Python 3.11 环境，但 alpha 和 py 必须使用同一个解释器与 NumPy ABI。

## 3. 用最新 alpha 源码编译 pybind SO

不要复制其他机器的 `_core.so`。必须在目标机器上从所选 alpha 提交重新编译。

CUDA 版本：

```bash
cd "$TT_ROOT/teletubbies-alpha"

cmake -S . -B build-cuda \
  -DPython3_EXECUTABLE="$PYTHON" \
  -DTELETUBBIES_ALPHA_BUILD_PYTHON=ON \
  -DTELETUBBIES_ALPHA_BUILD_TESTS=ON \
  -DTELETUBBIES_ALPHA_BUILD_EXAMPLES=ON \
  -DTELETUBBIES_ALPHA_BUILD_CUDA=ON

cmake --build build-cuda --target _core -j 8
```

无 CUDA 的验证机使用 CPU build：

```bash
cd "$TT_ROOT/teletubbies-alpha"

cmake -S . -B build \
  -DPython3_EXECUTABLE="$PYTHON" \
  -DTELETUBBIES_ALPHA_BUILD_PYTHON=ON \
  -DTELETUBBIES_ALPHA_BUILD_TESTS=ON \
  -DTELETUBBIES_ALPHA_BUILD_EXAMPLES=ON \
  -DTELETUBBIES_ALPHA_BUILD_CUDA=OFF

cmake --build build --target _core -j 8
```

设置运行路径。CUDA 和 CPU build 二选一，不要同时放入 `PYTHONPATH`：

```bash
export ALPHA_BUILD="$TT_ROOT/teletubbies-alpha/build-cuda"
export PYTHONPATH="$ALPHA_BUILD/python:$TT_ROOT/teletubbies-alpha/python:$TT_ROOT/teletubbies-py/src"
```

验证 Python 确实加载刚编译的 SO：

```bash
"$PYTHON" -c "import teletubbies_alpha as ta; import teletubbies_alpha._core as c; print(ta.__file__); print(c.__file__); print('cuda_devices=', ta.cuda_device_count())"
```

输出中的 `_core` 必须位于当前 checkout 的 `build-cuda/python/teletubbies_alpha/`。CUDA
build 成功但 `cuda_devices=0` 表示当前进程看不到 GPU；这不等于 CPU 流式因子不可用，但
不能据此声称 CUDA 路径通过。

## 4. 准备交付模型

解压开发验证包：

```bash
export DELIVERY=/path/to/lgbm1h_mlp4h8h_dev_validation_20260918
cd "$DELIVERY"
sha256sum -c PACKAGE_SHA256SUMS
```

两套版本均包含 1111-feature LGBM、两个 EMA ONNX 和策略配置：

```text
lgbm_ret1h_2025h1.txt
mlp_ret4h_ema_2025h1.onnx
mlp_ret8h_ema_2025h1.onnx
portfolio_2025h1.yaml

lgbm_ret1h_2025h2.txt
mlp_ret4h_ema_2025h2.onnx
mlp_ret8h_ema_2025h2.onnx
portfolio_2025h2.yaml

alpha_inventory_1111_strict.yaml
symbol_manifest_2025h1.json
symbol_manifest_2025h2.json
```

`2025h1` 训练区间为 `2024-11-30..2025-06-30`；`2025h2` 为固定参数 rolling train，
训练区间扩展到 `2025-12-31`，固定 OOS 是 `2026-01-01..2026-01-31`。

## 5. Canonical indicator 输入

因子 runner 不直接读取 target，也不读取未来 VWAP。输入是 framework 生成的 canonical
5 分钟 indicator parquet：

```text
/data/indicator/prod_v1_bar_5m/concat/2024-08-01_2026-01-31.parquet
/data/indicator/prod_v1_bar_5m/concat/2024-08-01_2026-01-31.metadata.json
```

要求：

- `timestamp_us` 是 bar 结束时间 `B`。
- 当前 bar 字段只使用 exchange timestamp 位于 `[B-5m,B)` 的事件。
- 每天 288 bars，`ti=0..287`。
- metadata 的 symbol 顺序和字段顺序为权威顺序。
- `/data/crypto` 只读；生成数据写到 `/data/features`。

如果需要从 framework 日文件重新构建 canonical parquet，先按 framework 的
`feature-dump/build_indicator_bar_5m.py` 生成日文件，再运行：

```bash
cd "$TT_ROOT/teletubbies-py"
teletubbies-bar-concat \
  --input-dir /path/to/framework/daily_indicator \
  --output /data/indicator/prod_v1_bar_5m/concat/rebuilt.parquet \
  --metadata-output /data/indicator/prod_v1_bar_5m/concat/rebuilt.metadata.json
```

## 6. 生成 1111 因子

从现有配置复制一份本机配置：

```bash
cd "$TT_ROOT/teletubbies-py"
cp configs/bar_signal_runner_alpha_all_1111_full_npy.yaml /tmp/alpha_1111_npy.yaml
```

在 `/tmp/alpha_1111_npy.yaml` 中确认：

- `expressions_file` 和 `input_columns_from` 指向
  `$DELIVERY/alpha_inventory_1111_strict.yaml`。
- 输入 parquet 和 metadata 指向当前机器的 canonical 数据。
- 输出根目录位于 `/data/features`。
- `execution=streaming`、`output_normalization=cszscore`、`output_clip_abs=20`。
- `valid_column=valid`、`reset_on_day=false`、`lookback=864`。

运行：

```bash
cd "$TT_ROOT/teletubbies-py"
OMP_NUM_THREADS=8 "$PYTHON" -m teletubbies_py.tools.bar_signal_runner \
  --config /tmp/alpha_1111_npy.yaml
```

输出契约：

```text
factors.npy   float32 (T,N,1111)
valid.npy     uint8   (T,N)
timeline.npy  int64   (T,4), columns=(td,ti,di,timestamp_us)
manifest.json
```

必须检查 `manifest.json` 中的 factor/symbol/timeline hash、`warmup_bars` 和 target alignment。
训练与评估不得使用 `warmup_bars` 之前的行。

## 7. 三模型推理与信号融合

模型输入是每个截面的 `features: float32[N,1111]` 和 `valid[N]`。两个 ONNX 的输入名为
`features`，输出名为 `prediction`；LGBM 的 `max_feature_idx=1110`。

在离线或在线配置中设置：

```yaml
class: teletubbies_py.bar_model.ensemble:LgbmMlpEnsembleModel
params:
  lgbm_model_path: /absolute/path/lgbm_ret1h_2025h1.txt
  mlp_ret4h_path: /absolute/path/mlp_ret4h_ema_2025h1.onnx
  mlp_ret8h_path: /absolute/path/mlp_ret8h_ema_2025h1.onnx
  factor_inventory_path: /absolute/path/alpha_inventory_1111_strict.yaml
  symbol_manifest_path: /absolute/path/symbol_manifest_2025h1.json
  num_threads: 8
  providers: [CPUExecutionProvider]
```

切换到 `2025h2` 时必须同时替换四个带版本的模型/manifest 文件，不能混搭。

融合公式固定为：

```text
z1 = CSZ(LGBM_ret1h_raw | valid)
z2 = CSZ(MLP_ret4h_raw | valid)
z3 = CSZ(MLP_ret8h_raw | valid)
ensemble_mean = (z1 + z2 + z3) / 3
meta_signal = CSZ(ensemble_mean | valid)
```

运行完整离线流式链路：

```bash
cd "$TT_ROOT/teletubbies-py"
systemd-run --user --scope \
  -p MemoryHigh=58G -p MemoryMax=62G -p MemorySwapMax=0 \
  taskset -c 0-15 env PYTHONPATH="$PYTHONPATH" OMP_NUM_THREADS=16 \
  "$PYTHON" -m teletubbies_py.tools.run_offline_streaming_pipeline \
  --config /path/to/local_offline_selected_pipeline_5m.yaml
```

该 runner 只输出因子、模型信号和 target weight，不执行 posterior fill、未来成交额限制或 PnL。

## 8. 重新训练

训练数据是第 6 节生成的 mmap NPY。Target parquet 统一采用：

```text
ret_h[ti] = log(vwap[ti+1+h] / vwap[ti+1])
```

即信号在 `ti` 结束时产生，下一根 bar 的 VWAP 是收益分母。模型特征中不包含这个未来
VWAP。MLP 使用 raw target 的 0.5%/99.5% winsorize、CS-demean、CCC loss、batch 16、
`1111->256->128->1`、dropout 0.3、lr 5e-4、EMA 0.999、1 epoch。LGBM 使用 50 rounds、
512 leaves、lr 0.05、feature/bagging fraction 0.9、max_bin 63。

基础入口：

```text
teletubbies-alpha/scripts/train_mlp_npy.py
teletubbies-alpha/scripts/train_lgbm_npy.py
```

严格 train/OOS 隔离时，固定收益 target 的 purge 应包括 next-bar entry：

```text
ret15m=4, ret1h=13, ret2h=25, ret4h=49, ret8h=97
```

历史交付模型使用的 MLP purge 是 ret4h=48、ret8h=96；这会让最后一个训练截面的 exit
触及 OOS 第一根 bar。若目标是逐字节复现历史 artifact，应保留历史值；若目标是新训练的
严格隔离版本，应使用 49/97，并视为新模型版本。

## 9. 最小验收

至少执行：

```bash
cd "$TT_ROOT/teletubbies-py"

"$PYTHON" -m pytest -q \
  tests/test_alpha_expression_streaming.py \
  tests/test_model_ensemble_contract.py \
  tests/test_model_replay.py \
  tests/test_offline_streaming_pipeline.py \
  tests/test_portfolio_solver_parity.py
```

有完整本地数据时，再运行：

```bash
OMP_NUM_THREADS=8 "$PYTHON" scripts/validate_selected_model_parity.py \
  --sample-count 32 --num-threads 8 \
  --output /tmp/selected_model_parity.json
```

验收标准：

- factor 顺序、symbol 顺序和 schema hash 完全相同。
- mask mismatch 为 0。
- 模型与保存信号最大绝对误差 `<1e-4`。
- batch、streaming 和 endpoint 测试通过。
- CPU、RSS 不超过机器资源的一半；不要一次性 `np.load` 完整 `(T,N,F)`。

## 10. 常见失败

- `_core` import 到旧路径：确认 `build-cuda/python` 位于 `PYTHONPATH` 最前面。
- `GLIBCXX` 错误：Python 环境和编译器运行库不一致，重新在目标环境编译 `_core.so`。
- CUDA build 但设备数为 0：检查 `nvidia-smi`、`/dev/nvidia*` 和进程环境。
- ONNX feature mismatch：使用了错误的因子清单或模型版本。
- symbol hash mismatch：canonical metadata、模型 manifest 和运行时 universe 顺序不一致。
- OOM：保持 NPY mmap、按 day/chunk 推理，并限制 `OMP_NUM_THREADS` 和 cgroup RSS。
