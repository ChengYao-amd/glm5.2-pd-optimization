# GLM-5.2 baseline goodput sweep

在本仓库 Dockerfile 构建的容器内运行，沿用 `../env.bashrc` 的配置和
`../run_profiles/fake_decode.patch`。默认 TP4/DP4/EP1、GPU 0–3、MXFP4 模型、
FP8 KV、mem-fraction 0.85、ISL10000/OSL500、EAGLE 5/6/topk1、模拟接受长度 3.61。
这里做性能测试和统计校验；fake KV / simulated acceptance 不用于模型准确率评测。

```bash
# 容器终端 1：前台启动，Ctrl-C 停止；CONC 是 server 容量，默认 64
bash /run/baseline_sweep/up.sh

# 容器终端 2：自动等待 ready，依次测 1 2 4 8 16 32 40 64
OUT_DIR=/run/baseline_sweep/results/baseline bash /run/baseline_sweep/sweep.sh

# 同一 server、同一目录追加边界复测，label 不能重复占用已有轮次
OUT_DIR=/run/baseline_sweep/results/baseline bash /run/baseline_sweep/sweep.sh \
  --label confirm --concurrencies 32 40 --repeats 2

# 单点测量；省略 OUT_DIR 时自动创建带时间戳的结果目录
bash /run/baseline_sweep/sweep.sh --concurrencies 32

# 根据已完成轮次重新汇总，不需要 server
python3 /run/baseline_sweep/analyze.py /run/baseline_sweep/results/baseline
```

`up.sh` 自动应用已有 fake-decode 补丁，清除共享配置里的 profile 设置，恢复 ROCm
默认 graph packet capture 行为；不调用 profiler。默认 decode graph buckets 是
`1 2 4 8 16 24 32 40 48 56 64`，按 server 容量截断并包含容量上限。
额外参数直接传给 SGLang，例如：

```bash
# 原 goodput 实验的并行配置（当前仓库默认是 DP4/EP1）
TP=4 DP=1 EP=4 bash /run/baseline_sweep/up.sh

# 扩大容量并指定 graph buckets；需要停止旧 server 后启动
CONC=128 bash /run/baseline_sweep/up.sh --cuda-graph-bs-decode 1 16 32 64 128
```

每点请求数为 `max(128, 8×并发)`，向上取整为并发的整数倍；预热 `max(16, 并发)`
个请求，原生客户端预热输出 32 tokens。可用 `--min-requests`、`--waves` 调整请求数，
用 `ISL`、`OSL` 调整长度。原始 160 条 synthetic prompts 在本地生成并循环扩展，
不下载数据集。`--ready-timeout` 默认 3600 秒，`--benchmark-timeout` 每点默认 2400 秒。

`client_details.patch` 只给原生 benchmark 增加逐请求耗时、成功状态和 SSE chunk
gap/token count 的保存；补丁应用到结果目录内的副本，不修改安装的客户端。
它与 Dockerfile 固定的 SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a` 配套。
每轮检查请求数、成功状态、输入/输出长度，并从逐请求数据重算 TPOT/ITL，
与原生统计交叉核对；检查失败立即停止，日志和原始数据保留。

每个结果目录包含配置 `config.json`、server 信息、客户端副本，以及：

- `rounds/<label>_c<并发>_r<轮次>/benchmark.jsonl`：原始结果；`benchmark.log`：客户端日志。
- 同轮 `requests.csv`、`metrics.json`：逐请求速率、TTFT/E2E/TPOT/ITL 分位数、达标比例和 goodput。
- `analysis/metrics.csv`、`analysis/summary.json`：全部成功轮次和最大已测达标并发。

每请求 decode 速率为 `(output_tokens - 1) / (latency - TTFT)`；70/80 tokens/s
分别对应 TPOT ≤14.2857/12.5 ms。Goodput 是达标请求的输出 token 总数除以测量时间。
P50、P90 和至少 90% 请求达标是三个独立规则，同并发的所有复测轮次均需满足。
汇总会列出已开始但未通过校验的轮次，对应并发不参与最大达标并发的选择。
原生 ITL 将 SSE chunk gap 均摊到新增 token；真实 chunk gap 单独保留。
TTFT/E2E 不含客户端等待并发槽的时间；`accept_length` 是 server 生命周期累计值。
并发是客户端上限，不保证实际 batch 恒定；server 日志在启动终端，需要时可用 `tee` 保存。
更换 server 配置或输入/输出长度后使用新结果目录，避免混合比较。

统计逻辑测试（无需 GPU）：

```bash
python3 -B -m unittest discover -s /run/baseline_sweep -p 'test_*.py'
```
