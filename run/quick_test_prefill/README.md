# GLM-5.2 quick_test_prefill

单机固定 batch 的真实权重 prefill 测试。复用镜像内 SGLang 的 `one_batch`
模型加载、DP 同步和 `extend`，只加载 target 模型。输出 prefill 延迟、输入吞吐、
Perfetto trace 和算子/kernel CSV；不需要启动 HTTP server。

## 运行

在有 8 张空闲 GPU 的机器上执行（已在 `xiaobche@crsuse2-m2m-137` 验证）：

```bash
cd /shared_nfs/yaoc/work/infera-test/glm5.2-pd-optimization/run/quick_test_prefill
bash start_container.sh

# 默认：全局 batch=8，每请求 32768 tokens，预热 2 次、计时 5 次。
docker exec glm52-quick-prefill bash /quick_test_prefill/bench.sh

# 性能测量后额外采集一次未命中部分的 prefill；形状信息按需开启。
docker exec glm52-quick-prefill bash /quick_test_prefill/profile.sh --profile-record-shapes

# 小规模检查 / 长输入分 chunk / 每个 rank 的 trace。
docker exec -e ISL=1024 -e STEPS=2 glm52-quick-prefill bash /quick_test_prefill/bench.sh
docker exec -e ISL=70000 glm52-quick-prefill bash /quick_test_prefill/profile.sh
docker exec -e PROFILE_RANKS=all glm52-quick-prefill bash /quick_test_prefill/profile.sh

# 每请求 75% 的输入已有 GPU prefix KV；32K 中只计时剩余 8K 的 prefill。
docker exec -e ISL=32768 -e CACHE_HIT_RATE=0.75 glm52-quick-prefill bash /quick_test_prefill/bench.sh
# 也可使用命令行参数；profile 同样只采集剩余 token 的计算。
docker exec glm52-quick-prefill bash /quick_test_prefill/profile.sh --cache-hit-rate 0.75

# 容器外也能预览参数，无须安装 torch/SGLang。
bash bench.sh --dry-run

# 使用结束后释放容器；本机结果和 AITER cache 保留。
docker stop glm52-quick-prefill
docker rm glm52-quick-prefill
```

默认 `WORKSPACE_DIR=/tmp/glm52-quick-prefill-$USER`，由启动脚本传入容器；
137 的 `/shared_nfs` 为只读，所以默认写入节点本地磁盘。
`NAME`、`WORKSPACE_DIR` 可在 `start_container.sh` 前设置。
容器已存在时直接报错，不会停止其他任务。
首次运行需要加载权重并编译缺失的 kernel，可能花几分钟；后续复用本机缓存。

| 变量 | 默认值 / 含义 |
|---|---|
| `TP / DP / EP` | `8 / 8 / 1`，DP 为 attention DP；TP 和全局 batch 必须能被 DP 整除 |
| `BATCH_SIZE`（或 `CONC`） | `8`，全局请求数，每个 DP shard 分到 `BATCH_SIZE / DP` 个 |
| `ISL` | `32768`，每请求完整输入长度，包含命中的前缀 |
| `CACHE_HIT_RATE` | `0`，每请求缓存 token 比例，范围 `[0, 1]`；按 KV page 向下对齐，实际比例写入结果 |
| `CHUNK_SIZE` | `32768`，SGLang 原始参数；DP8 下解析为每个 DP shard `4096` tokens |
| `WARMUP_STEPS / STEPS` | `2 / 5`，每次重建指定长度的真实 prefix KV，再测剩余输入 |
| `PROFILE_STEPS / PROFILE_RANKS` | `1 / 0`；rank 支持 `0,1` 或 `all` |
| `MEM_FRACTION / KV_CACHE_DTYPE` | `0.85 / fp8_e4m3` |
| `RESULTS_DIR / OUT_DIR` | `$WORKSPACE_DIR/results` / 其下新的时间戳目录；已有 OUT_DIR 会报错 |

用 `docker exec -e KEY=value ...` 调整运行参数。脚本后的参数可覆盖 Python 参数，
未知参数交给 SGLang；例如 `--input-len 16384`、`--profile-with-stack`。
改变 GPU 数时同时设置 `HIP_VISIBLE_DEVICES`、`TP` 和 `DP`。

## Cache hit rate 的口径

`CACHE_HIT_RATE` / `--cache-hit-rate` 控制**每个请求的 GPU 驻留 prefix KV token 比例**。
所有请求使用相同命中长度，不表示“多少比例的请求发生过命中”，也不模拟缓存查找、
淘汰或 HiCache 从 CPU/存储加载 KV。即使禁用 radix cache，这个局部计算模式仍然生效：
脚本直接保留真实 forward 生成的 KV 和 DSA indexer 状态，并传给后续 `extend`。

设完整长度为 `L`，配置比例为 `r`，KV page size 为 `P`，实际命中长度为：

```text
cached = floor(min(floor(L × r), L - 1) / P) × P
new = L - cached
effective_cache_hit_rate = cached / L
```

保留至少一个未缓存 token 来生成 next-token logits；设为 `1` 时仍需计算末尾不足一页
或一整页，因此实际比例可能小于配置值。较短前缀对齐后也可能为 0。
`--dry-run` 只展示目标比例；实际 KV page size 和命中长度在模型加载后确定。

每轮先清空池，在计时和 profiler **之外**计算 `[0, cached)`，然后只测
`[cached, L)`。例如 `ISL=32768, CACHE_HIT_RATE=0.75`，在页大小允许精确对齐时，
已有 24576 tokens 的 KV，计时计算 8192 个新 tokens；attention 仍读取已有的长上下文。
不能用 `ISL=8192, CACHE_HIT_RATE=0` 替代该场景。

对齐真实业务时，还需匹配完整 ISL、新 token 长度、batch、chunk budget、并行配置和
KV dtype。这个模式适合分析缓存已在 GPU 时的局部计算成本；线上混合长度/命中率、
动态调度、HiCache 加载/写回和端到端 TTFT 应继续用服务端 bench 验证。
相同平均命中率、不同请求分布，性能也可能不同。

## 环境与测量口径

镜像和计算配置对齐
[`2p1d-sweep`](../../../../GLM-5.2-pd-opt/Infera-isl-debug/yaocheng/2p1d-sweep/README.md)：

- 镜像 `infera-sglang:v0519-yihou-0917-nextnfix-hicache`，启动时核对 ID
  `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35`。
  SGLang commit 为 `7ccbf5fd04f7ee23095fc38e49e749d58dc18282`，不额外打补丁。
- 模型 `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`；DSA `tilelang`，
  默认 top-k、fused indexer、fused QK norm/RoPE、AITER all-reduce fusion，
  `HSA_NO_SCRATCH_RECLAIM=0` 和 `env.sh` 中的运行变量均沿用参考。
- 这是计算微基准：无 scheduler、HiCache 加载/写回、KV 传输、路由或 decode/MTP。
  使用 eager prefill（参考实跑的 prefill/decode graph backend 也均为 disabled）；
  结果不等同于线上 TTFT 或 2P1D 吞吐。
- 每个 DP shard 使用不同随机 token；attention TP 副本使用相同输入。
  请求均匀分配；长输入按页对齐、等长 chunk 推进，前一 chunk 的真实 KV 保留。
  采用 SGLang 解析后的 chunk budget：默认每请求 4096 tokens/chunk，0 命中时共 8 chunks，
  75% 命中时计时 2 chunks。
  这套固定分块规则不模拟服务端动态调度。
- 计时包含 batch 准备、未命中部分各 chunk 的 forward、采样和设备同步，排除权重加载、
  JIT/预热、随机请求生成、池重置、prefix KV 生成和测量外的 rank 同步。
  每次取所有 rank 的最大耗时。完整输入吞吐 = `全局 batch × ISL / 平均耗时`，
  新 token 吞吐 = `全局 batch × new / 平均耗时`；每 GPU 吞吐再除以 TP，避免重复计算 TP 副本。
  命中前缀计入完整输入吞吐，因此比较计算效率时应查看新 token 吞吐。
- profiling 在干净计时结束后进行，因此同一次 `profile.sh` 的 `result.json`
  仍是未加 profiler 的性能数据。trace/CSV 仅覆盖后续 profile 次数。

## 输出

- `result.json`：全局逐次延迟、平均/中位延迟、完整输入和新 token 的 token/s 及 token/s/GPU；
  `requested_cache_hit_rate` / `effective_cache_hit_rate`、`cached_tokens_per_request` /
  `new_tokens_per_request`、`kv_page_size` 记录目标及实际形状，`chunks_per_request` 仅计未命中部分。
- `rank_N.json`：每 rank 原始耗时；`status.json` 的 `exit_code=0` 表示整次运行成功。
  profiling 失败时即使已有 `result.json`，状态仍为失败，命令也返回非零。
- `config.json`、`server_args.json`、`command.txt`：实测环境、解析后的 SGLang 参数和命令。
- `traces/prefill-TP-N.trace.json.gz`：可直接导入 Perfetto。
- `traces/operators_rank_N.csv`：PyTorch 算子次数、CPU/GPU 耗时和可选输入形状。
- `traces/kernels_rank_N.csv`：按 GPU 自身耗时降序的完整 kernel 表，单位微秒。
  算子表的 GPU 时间包含下层 kernel，不能与 kernel 表相加；多 stream kernel
  可能重叠，kernel 时间之和也不等于批次延迟。
- `PROFILE_STEPS > 1` 时每次采集单独放入 `traces/step_1/`、`traces/step_2/` 等目录，
  每份 trace/CSV 仅包含该次剩余输入的 prefill，排除两次采集之间重新生成 prefix 的计算。

## 实测验证

2026-09-22，`crsuse2-m2m-137`，8 × MI355X，TP8/DP8/EP1，原始 chunk 参数 32768：

| 全局 batch | ISL | chunks/请求 | 预热/计时次数 | 平均延迟（秒） | 输入 token/s | 输入 token/s/GPU |
|---|---|---|---|---|---|---|
| 8 | 1024 | 1 | 2/2 | 0.1593 | 51,412.16 | 6,426.52 |
| 8 | 32768 | 8 | 2/5 | 7.1796 | 36,512.22 | 4,564.03 |
| 16 | 70000 | 35 | 1/2 | 31.2593 | 35,829.38 | 4,478.67 |

三组均额外 profile 1 次；前两组采集 rank 0 和输入形状，70K 采集全部 8 个 rank。
32K trace 包含 20,983 次 GPU kernel 调用、76 类 kernel、576 组算子/形状。
70K 覆盖每个 DP shard 两个请求及最后 368 tokens 的非整页 chunk。
短输入和 70K 点用于功能验证，计时样本较少。
原始 JSON、CSV、trace 和日志保存在
[`workspace/quick_test_prefill/validation-20260922`](../../workspace/quick_test_prefill/validation-20260922/)，
由仓库现有的 `/workspace/` 规则忽略。
