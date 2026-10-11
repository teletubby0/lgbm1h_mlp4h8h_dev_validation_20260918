# LGBM1h + EMA MLP4h/8h 开发验证包

生成日期：2026-09-18

本包包含当前选中三模型 ensemble 的 2025H1 和 2025H2 rolling-train 两套模型、策略参数、
1111 因子清单、symbol manifest、复现文档和校验文件。

## 1. 先校验包

```bash
sha256sum -c PACKAGE_SHA256SUMS
```

所有文件必须显示 `OK` 后再运行。

## 2. 模型和策略

### 2025H1

训练区间：`2024-11-30..2025-06-30`

```text
lgbm_ret1h_2025h1.txt
mlp_ret4h_ema_2025h1.onnx
mlp_ret8h_ema_2025h1.onnx
portfolio_2025h1.yaml
symbol_manifest_2025h1.json
```

### 2025H2

训练区间：`2024-11-30..2025-12-31`

固定 OOS：`2026-01-01..2026-01-31`

```text
lgbm_ret1h_2025h2.txt
mlp_ret4h_ema_2025h2.onnx
mlp_ret8h_ema_2025h2.onnx
portfolio_2025h2.yaml
symbol_manifest_2025h2.json
```

共同因子清单：

```text
alpha_inventory_1111_strict.yaml
```

两个 ONNX 接收 `features: float32[sample,1111]`，输出 `prediction: float32[sample]`。
LGBM 同样使用 1111 个特征，`max_feature_idx=1110`。

## 3. Ensemble 契约

```text
z_lgbm = valid-only CSZ(LGBM ret1h raw)
z_mlp4 = valid-only CSZ(EMA MLP ret4h raw)
z_mlp8 = valid-only CSZ(EMA MLP ret8h raw)

ensemble_mean = (z_lgbm + z_mlp4 + z_mlp8) / 3
meta_signal = valid-only CSZ(ensemble_mean)
```

不能直接平均三个 raw prediction，也不能省略最后一次 CS-zscore。

## 4. 代码版本

本包验证对应：

```text
teletubbies-alpha     24e3e6c6774e02dec5b2e8f2206d4f774f70075d
teletubbies-py        c918a5fc0cb8d18283f7d87c6b1c780f651d25b5
teletubbies-framework 64019fccb75823b47e90106ca104d7b1838ce1af
```

生成本包时，三个本地 `main` 均与刷新后的 `origin/main` 相同。但是：

- `teletubbies-py` 的 `docs/reproduce_selected_pipeline_zh.md` 是本次新增文档，随包提供，尚未提交到上述 commit。
- `teletubbies-alpha/scripts/run_rolling_train_2025_combo.py` 是当前 H2 rolling-train 编排脚本，随包放在 `source_helpers/`，尚未进入上述 `origin/main`。

因此不能声称三个仓库的所有最新文档和 rolling-train orchestration 已全部 push；核心已提交版本和
本包附加文件必须一起使用，或者先将这些改动审核后提交。

## 5. 从零复现

严格按以下文档执行：

```text
docs/reproduce_selected_pipeline_zh.md
```

它包含：

1. checkout 三个固定仓库版本。
2. 用同一个 Python 3.11 环境从 alpha 源码编译 pybind `_core.so`。
3. 验证 Python 实际加载新编译 SO。
4. 从 framework canonical indicator 生成 `(T,N,1111)` 因子。
5. 运行三个模型以及两级 CS-zscore ensemble。
6. 重新训练和 parity 验收入口。

不要把包内模型文件当作 alpha pybind SO。`_core.so` 与本机 Python/C++/CUDA ABI 相关，必须
在验证机上用最新 alpha checkout 重新编译。

## 6. 本机验证记录

本包生成前已执行：

- 使用 `rainbow` Python 3.11.11 重新配置并编译 alpha CUDA pybind `_core`。
- `_core.so` SHA256：`830897b3a41a5c4582d9e6c76bc959832c44a7ed0342f80d30c373aa6257402c`。
- teletubbies-py 核心测试：17 passed。
- 8 个完整 OOS 截面的模型 parity：通过。
- 最大分量误差：`1.0430813e-7`。
- 最终信号最大误差：`2.8610229e-6`。
- mask mismatch：0。

记录位于：

```text
validation/selected_model_parity_latest_alpha.json
```

当前执行环境的 CUDA build 成功，但进程报告 `cuda_device_count=0`，所以这里只确认 CPU
流式/模型 parity；没有把 CUDA 设备运行声明为通过。

## 7. Target 时间口径

搜索和训练统一使用：

```text
ret_h[ti] = log(vwap[ti+1+h] / vwap[ti+1])
```

即 `ti` 结束时产生信号，下一根 bar 的 VWAP 是收益分母。详细字段时间口径见：

```text
docs/indicator_field_timestamp_map_2026-09-18.csv
```

## 8. 原始来源路径

生成本包时的规范交付目录：

```text
/data/features/alpha/deliveries/lgbm1h_mlp4h8h_delivery_2025h1_2025h2
```

2025H1 完整 release：

```text
/data/features/alpha/releases/lgbm1h_mlp4h8h_selected_2025H1
```

2025H2 rolling artifacts：

```text
/data/features/alpha/models/lgbm1h_mlp4h8h_rolling_train20241130_20251231_oos20260101_20260131_v1
```

策略 YAML 中的绝对路径用于 provenance；换机器时应按本包文件位置改成本机绝对路径。
