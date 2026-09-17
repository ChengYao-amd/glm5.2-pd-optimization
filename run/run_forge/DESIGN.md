# run_forge 方案草案

状态：**待 review，下面的脚本、配置和命令接口是拟议实现，尚不可执行。**
检查日期：2026-09-14。输入：`workspace/kernel-analyze-poc/025/analysis`。

建议先以 **TP4 all-reduce** 跑通任务准备、原实现 baseline、优化和导出，再做
**residual/RMSNorm/DP staging**。使用 `kernel-agents forge-loop`，在本目录实现一层
任务准备及运行脚本；继续使用已有容器、Slurm 和 serving benchmark 入口。

## 1. 025 结果检查

在 Slurm allocation 136652、节点 `crsuse2-m2m-025` 的现有
`kernel-analyze-poc-025` 容器中重新执行了 `validate_handoff.py`：9 个任务全部通过。
本地比较了全部 `drafts/*.yaml` 与 `tasks/*.yaml`，当前内容一致。

| 候选 | 稳定窗口 kernel 时间占比 | 当前状态 | 首轮处理 |
|---|---:|---|---|
| `tp4_bf16_allreduce_001` | 11.3336% | ready | 修正下面的 ABI 描述后准备 driver；先跑 |
| `residual_norm_quant_chain_001` | 5.0131% | ready | 明确算子链和数值语义后准备 driver；第二项 |
| `mxfp4_moe_chain_001` | 36.8032% | needs_evidence | 补每层 routing IDs/weights、expert occupancy、有效 sorted blocks |
| `sparse_mla_attention_chain_001` | 36.6953% | needs_evidence | 补 active KV lengths、page table、selected-index locality |
| `dsa_indexer_chain_001` | 8.9440% | needs_evidence | 补 active seqlens、page/index 分布、有效 logits 列数 |
| dense/draft GEMM、NCCL、async HtoD、metadata tail | — | not_recommended，4 项 | 保留原因，不提交优化 |

占比分母为十次稳定 `TARGET_VERIFY bs=8` × 四 rank 的 kernel duration 总和
**1,633,884.016 us**，八个互斥 family 的时间之和与该值一致；百分比舍入后合计
100.0001%。这些比例不等于 e2e 可节省时间。稳定 attention 是本地 `[48,6144]`，
DP gather 后 MLP/MoE/all-reduce 是全局 `[192,6144]`；MoE 的 `M=256` 是 tuning key
padding，不能作为有效行数。

`run.json.status=partial` 来自分析阶段超时，三个阶段和任务发布均已完成；不应一律
拒绝这个目录。下游应按重新验证的每任务 readiness、输入完整性和 driver gate 决定。
`ready_for_handoff` 表示可开始准备 driver；025 没有独立 driver、microbenchmark
baseline 或已校准的数值阈值。

### 提交前要修正/补全的内容

1. **All-reduce 的算子 ABI 与调用侧写回要分开。**
   已读原容器 `/aiter/aiter/dist/device_communicators/custom_all_reduce.py:1177-1239`：
   `all_reduce(inp, out=None)` 分配独立 output，`custom_all_reduce()` 也按 out-of-place
   路径返回结果。SGLang `/sglang/python/sglang/srt/layers/dp_attention.py:495` 的
   `global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)` 才提供调用侧写回。
   当前 handoff 的 `aliasing: in_place_collective_output` 混合了这两层语义；不能据此
   构造 input/output 必须 alias 的 raw-kernel driver。

2. **All-reduce 模板参数有误标。**
   `/aiter/csrc/include/custom_all_reduce.cuh:483` 的第三个模板参数是
   `is_broadcast_reg_outptr=false`，不是 task 中的 `accumulate_template: false`。
   同一实现第 527–548 行仍以 FP32 累加并转回 BF16。recipe 应覆盖这个字段，保留
   原始 task 和修正依据；不能把它解释成不累加或非 SUM。

3. **Norm 的 reference 需要重新按 GLM/AITER 语义定义。**
   实例是 `FUSE_QUANT=false`，保留两个独立 BF16 输出，不生成 scale。
   从原模型 `config.json` 补到了 `rms_norm_eps=1e-5`。AITER
   `rmsnorm_quant_kernels.cu:100-150` 先在 FP32 做 residual add，写出 BF16 residual，
   同时继续用尚未舍入的 FP32 sum 计算 norm。不能直接照搬仓库 Gemma driver 的
   BF16 add 和 `(1 + weight)` 公式。GLM reference 应计算 FP32 sum 的 RMSNorm，
   应用原 weight，再分别转换 normalized output 和 residual output；用原实现校准容差。

4. **报告开头两项 union 指标的名称不准确。**
   41.887 ms 对应 graph 关联的全部 GPU events 的区间并集，含 copy/memset；
   419.088 ms 对应 graph 起止 spans 的跨 rank 并集。均不能标作纯 kernel busy-time。
   依据为 `scratch/summarize_traces.py:155-180` 和
   `scratch/build_aux_evidence.py:136-144`；核心 family 排名分母不受此影响。
   all-reduce 的 80.012/120.017 GB/s 已正确改成 calculated/estimated，不能再用于
   宣称实测 XGMI 利用率。

5. **原生 program 渲染不能完整交接。**
   当前 `generate_program_md()` 没有渲染 `x-handoff`、`kernels_to_review`、
   `validation_gates` 和 validation shapes，还写有单文件修改规则。只执行
   `kernel-agents program task.yaml` 会丢掉关键输入/副作用信息，并与多文件任务冲突。
   这里应自建完整 renderer，输出修正后的语义、cases、源码关系和可编辑范围。

这次只新增本方案文档，原始 analysis/task/evidence 保留。后续 prepare 产出
`resolved_task.yaml` 和 `errata.json`，记录上述修正、源码位置和原 task hash。

## 2. KernelForge 接口选择与边界

核对版本：本地 KernelForge `cd9c5850699b0550c2aa06be83c3645cf4e98e24`。
容器的 `/KernelForge` 不带 Git 元数据，但其 `cli.py`、`task_preparer.py`、`program.py`
SHA256 与本地该版本相同；完整源码/依赖身份仍由正式 preflight 建 manifest。

- `kernel-agents run task.yaml` 是通用 orchestrator；本流程选 `forge-loop`，直接固定
  kernel、driver、case 合约、workspace 和恢复语义。
- `forge-loop` **没有直接接收 handoff YAML 的参数**。需要 `--kernel`、`--driver`、
  `--program-md-file`，推荐同时传 `--invocation-spec-file`。
- `--prepare-task` 可创建/修复 driver，且禁止改 kernel/source；仍需我们先准备真实源码、
  public callable、输入契约和固定 reference。它不会凭空恢复动态路由或 KV 分布。
- 上游没有独立的 `--prepare-only` CLI。为了能在优化前检查 driver，`prepare.py`
  通过一个小适配层调用当前版本的 `preflight_task` / `prepare_task_sync`，复用上游准备
  及回滚逻辑。适配层需要版本检查；不另写一套优化 agent，也不修改上游。
- `--nproc-per-node 4` 要与 self-launching driver 配合；该参数本身不是给任意单进程
  driver 自动添加 distributed 初始化。driver 无 `RANK` 时启动 torchrun，有 `RANK`
  时直接作为 worker，以兼容 KernelForge 的逐 rank profiling。
- `--fellow` 一次选一个。YAML 中 `[aiter, hip]` 不会在 forge-loop 中自动形成搜索矩阵。
  unsupported fellow 会 fallback，故下游必须先校验；TileLang 源码身份不等于存在
  `tilelang-fellow`。
- 当前 `--max-hours` 最小 1；`--max-iters` 只是兼容参数，不能限制迭代数。
  **`max_hours <= 2` 时 Analysis 阶段不采硬件 profiling**，即便传了 `--profiling`。
  想跑带硬件分析的首轮 campaign，建议每任务先给 3 小时，prepare 另限时 45 分钟。
- `--source-files` / `--target-functions` 是定位提示，不是 edit allowlist。
  实际修改范围、oracle/fixtures 完整性必须由本项目检查。

## 3. 建议定义的脚本

第一版实现前四个 Python 入口；第五个 e2e 脚本在有可导出 candidate 后实现。
公共代码放 `lib/`，各入口共享配置、身份检查和日志格式。

| 文件 | 职责和主要参数 | 输出/成功条件 |
|---|---|---|
| `check.py` | `--analysis-dir`、`--kernelforge`、`--config`；重验 handoff、打印 ready/blocked/skipped 队列；在源容器核对 source roots、版本、GPU/rank/graph 模式、依赖；`--driver-bundle` 可额外做 GPU driver 检查 | `check.json`；区分 metadata/source 检查与实际 GPU preflight |
| `prepare.py` | `--analysis-dir`、`--task-id`、`--output-dir`、`--config`；快照源码和证据、应用语义修正、定义 cases/reference、生成 program/spec，调用上游 driver preparer；`--materialize-only` 仅生成 bundle，便于检查 | 独立 Git workspace、`resolved_task.yaml`、`program.md`、`invocation_spec.json`、driver、`preflight.json`、`baseline.json` |
| `run.py` | `--bundle`、`--max-hours`；检查已准备任务，调用 forge-loop；`--dry-run` 打印脱敏配置和精确 argv，`--resume` 使用原 campaign；首版一次一任务 | 原生 campaign/checkpoints、`result.json`、日志、退出状态和 PID 记录 |
| `collect.py` | `--bundle`；读取原生结果和 Git diff，对 kept revision 做独立复验，核对 oracle/source/case 身份，导出按 repo 拆分的补丁及报告；`--report-only` 只汇总已有结果 | `summary.md/json`、各 case baseline/candidate、`aiter.patch` / `sglang.patch`、replay 命令、e2e 待验收清单 |
| `bench_e2e.sh`（第二阶段） | 对已导出的补丁，在专用 baseline/candidate 服务环境做重复 A/B；复用 `run_profiles/up.sh`、`bench.sh`、`profile.sh` | 无 profiler 的 A/B 数据、补丁加载证明、优化后短 trace、最终验收结论 |

不需要另外维护一套 `up.sh`：宿主侧继续使用 `run/start_container.sh` 和现有
`srun --jobid ... --nodelist ... --overlap`。正式实验使用新的 `EXP`、`WORKSPACE_DIR`
和唯一 `NAME`；已有 launcher 会 stop/rm 同名容器，外层应先拒绝名称冲突。
任务脚本在容器内运行，`/aiter`、`/sglang` 是输入源码路径，避免宿主机 validator
因容器路径不存在而误报。当前 allocation/node/container 只是本次检查位置，不硬编码到通用脚本。

辅助文件建议：

```text
run/run_forge/
  README.md
  config.example.yaml          # 默认 workload、路径、agent/runtime、预算
  check.py
  prepare.py
  run.py
  collect.py
  bench_e2e.sh                 # 第二阶段
  lib/
    forge_api.py              # 唯一耦合上游 prepare API 的位置
    bundle.py                 # snapshot、manifest、路径映射、program/spec 渲染
    runtime.py                # 环境、凭据加载、GPU 占用锁、子进程、脱敏日志
  recipes/
    tp4_bf16_allreduce.yaml    # 源码入口、修正、cases、fellow、可编辑范围
    residual_norm_staging.yaml
  templates/
    program.md
    allreduce/                # driver 起点、FP32 reference、distributed graph harness
    norm_staging/             # 双输出 reference、staging oracle、graph harness
```

`recipes` 是这两个已知算子的显式适配，不恢复按 kernel 名称自动猜 backend 的 taxonomy。
driver 在准备阶段由 KernelForge 补全；reference/cases 来自审定 recipe，准备阶段也需保护。

## 4. Workspace 与输入配置

这里的 Git workspace 是传给 `forge-loop --workspace` 的实验代码目录，用 Git 记录
baseline、每次候选修改和 best revision，并支持回退/恢复。可以从源码副本 `git init`
并提交初始版本，不要求新建远程仓库，也不特指 `git worktree`。

KernelForge 引擎本身不要求 SGLang/AITER 源码仓库；具体需要哪些源码和运行依赖由
任务的 kernel/driver 决定。第一版按任务准备：

| 任务范围 | 可编辑 workspace 内容 | 外部依赖/验证 |
|---|---|---|
| 独立 Triton/HIP kernel | kernel、driver、reference 和所需 helper | 对应 PyTorch/Triton/ROCm 工具链；通常不需要 SGLang/AITER |
| 本次 raw TP4 all-reduce | AITER 源码副本、driver、reference | AITER 编译/通信依赖与四 GPU；SGLang 可先只作调用语义参考和后续 e2e 验证环境 |
| 本次完整 norm/staging 边界链 | SGLang 与 AITER 的相关源码及构建依赖、driver | 两者的运行依赖；完整四 rank 链验证 |

一任务一独立 Git repo。跨 AITER/SGLang 修改时，建议统一把两者内容快照放到同一
repo 中，使 KernelForge 的 commit/revert 覆盖多文件改动；记录各来源 repo 的原
commit、dirty diff 及 submodule 内容身份。复制时不带嵌套 `.git`，不能用指向运行
环境的可写 symlink。需要 SGLang 快照的任务及 e2e 环境，必须包含这次 serving 的
六项既有 SGLang 文件差异，不能仅 checkout 原 commit。

```text
workspace/<forge-exp>/<task-id>/
  input/                       # 原 task、resolved task、errata、关联 evidence
  manifest.json                # 源 commit+diff/hash、镜像、依赖、workload、配置
  repo/                        # 只在这里 commit/revert
    aiter/                     # 固定来源快照
    sglang/                    # 任务涉及 SGLang 修改时加入
    measurement/
      driver.py
      reference.py
      graph_harness.py
      cases.json
      fixtures/
      program.md
      invocation_spec.json
    <KernelForge 原生 campaign 状态>
  artifacts/
    preparation/               # 包含上游 task_preparation 审计记录
    preflight.json
    baseline.json
    experiments/
    result.json
    logs/
  export/
    summary.md
    summary.json
    aiter.patch
    sglang.patch
```

构建/JIT cache 按 task 与 source fingerprint 隔离；运行 scratch 使用节点本地 `/tmp`，
不在 NFS 放 ROCm TMPDIR。原 025 cache 只作证据。进程启动时显式从候选快照加载
SGLang/AITER，验证 `module.__file__`、JIT include/build roots、加载 `.so` 和对应源码
fingerprint，防止实际执行 `/aiter` 的旧 build。仅改 `PYTHONPATH` 不足以证明换了实现。

共用配置建议包含：

| 配置组 | 建议值或规则 |
|---|---|
| 输入 | analysis dir、KernelForge root、`/aiter` / `/sglang` / model config 路径 |
| 设备/workload | gfx950、MI355X、TP4/DP4/EP1、本地 48 行/全局 192 行、H=6144；匹配 graph packet capture 模式 |
| Agent | 显式 provider/model/effort；可沿用 025 的 `codex` / `gpt-5.6-sol` / `max`，关闭自动跨 provider fallback |
| 预算 | prepare 45 分钟、campaign 3 小时/任务；均可覆盖；prepare 预算要扣除其自身 probe 时间 |
| Backend | all-reduce 先 `hip-fellow`；norm/staging 先 `triton-fellow`，任务类型 `repository` |
| 测量 | 固定 seed/cases/reference，`bench_repeat=3`；driver 必须支持 `--repeat` |
| 数值 gate | 准备阶段校准并写入 manifest；SNR、allclose、exact placement 和副作用检查组合；缺值时不能静默采用 30 dB |
| 知识库 | 首轮显式 `--no-experience-kb`，外部经验库读写作为后续配置项 |

gateway 挂载复用 `LLM_GATEWAY_DIR`，Python launcher 按已有分析 runner 的方式加载
`/llm_gateway/credentials.json` 到子进程环境，并对日志脱敏。仅挂载该文件不会自动让
KernelForge 获得所需环境变量；不在命令参数、manifest 或 bundle 中保存凭据。

## 5. 两个 driver 的具体契约

### 公共接口及 gates

| 调用 | 要求 |
|---|---|
| `python driver.py` | 完整 correctness suite；输出 `SNR: <数值> dB` 和 `allclose: True/False`；任一语义/数值检查失败时非零退出 |
| `python driver.py --bench-mode --warmup 10 --iters 30 --repeat 3` | 完整 benchmark suite；逐 case 输出 `case_ms: <稳定ID> <ms>`，另输出 `wall_ms:` 样本或 `median_ms:` / `mean_ms:`；全程 graph replay |
| `python driver.py --profile-run` | 只运行目标 case 路径，完成初始化/warmup后少量执行并同步；不夹带 reference/correctness/benchmark 输出 |
| `--profile-case <id>` | 精确选择单 case，供不同 shape/分布的 profiling 归因 |

`cases.json` 是本项目输入格式，由 driver 消费；Forge 不会自动读取它。
另外在 `invocation_spec.json` 中输出
`tests.driver_contract.case_selectors: [{"CASE_ID": "..."}, ...]`，并保留 invocation
参数、source locator 和 fixture 指针。上游 preflight 使用这些 ID 检查完整 case 集；
本项目再检查没有新增/漏掉的 scored case、分数角色和权重没有变化。

prepare 必须通过完整 correctness、benchmark 解析、真实 graph replay、profile contract，
并实测原实现的 baseline 和噪声。零填/dirty 输出后验证 replay 确实产生结果；准备后将
driver/reference/harness/cases/fixtures 固定并记录 hash。失败产物保留，但不进入优化。
baseline latency 从实际 driver 取得，不能把 trace 的 29.487 us 或约 4 us 填进去。

### TP4 all-reduce

- 原实现 anchor：`aiter/csrc/include/custom_all_reduce.cuh`，包含对应 `.cu`、header、
  Python communicator 和调用关系；GPU/rank 数固定为 4，BF16 `[192,6144]`，
  每 rank 输入 payload **2,359,296 B**。
- 借用上游 `examples/aiter-allreduce-forge-loop/driver.py` 的 rank 启动、IPC/capture、
  teardown 和最慢 rank 计时结构；替换原 Kimi/TP8/H=7168 默认 suite，不照搬性能结论。
- 三类 case：`gather_192x6144`（每 rank 仅自有 48 行非零）、
  `moe_out_192x6144`、`dense_out_192x6144`（各 rank 均有贡献）。后两者采用明确记录的
  可复现合成数值与极值用例，不能声称已采到真实激活分布。
- reference 用各 rank 输入做 FP32 SUM 后转 BF16，检查四 rank 全部输出、输入保持性、
  output alias 合约和可重复性；阈值按原实现校准。压力用例覆盖符号抵消、零值、大/小量级、
  连续 graph replay、信号状态和不同 rank 到达时序。
- raw operator 计时保留生产 capture/注册方式。边界链的诊断另含 gather 写回、下游
  scatter/norm；两种计时不可相互替代，新增转换或拷贝不能移到计时外。
- 计时在每 rank 用 GPU events，跨 rank 汇总 `max` 后由一个协调出口打印结果；
  correctness 汇总最差 rank。reference 和汇总 collective 放在目标计时区间外。
- 工作负载频率为 gather 79、MoE 输出 75、dense 输出 3，共 **157 次/graph/rank**。
  Forge 的等权分数之外，报告保留这三个频率及外部加权时间估计；边界链和 e2e 继续验证
  rank skew 与真实依赖是否改变了收益。

### Residual/RMSNorm/DP staging

- anchor 可选 SGLang `layers/dp_attention.py`；source files 同时覆盖 communicator、
  layernorm、memcpy Triton 和 AITER norm 实现。聚焦 norm 与 gather/scatter 边界，
  先固定 all-reduce 实现，避免把两个任务的收益混在一起。
- 本地 `[48,6144]`、全局 `[192,6144]`，contiguous/non-aliasing；
  `epsilon=1e-5`、BF16 双输出、`FUSE_QUANT=false`，采用上面的 FP32 reference。
- 性能 cases 覆盖 `norm_then_gather`、`scatter_then_norm`、final gather 及首/尾层特殊
  norm；同时给出完整 **78 层边界序列** 的 composite case。完整边界序列包含真实
  4-rank collective，不能拿 1-GPU mock 判定整个 family 通过。
- 允许 1-GPU component 模式做早期 debug，但必须标记为局部子任务，保留 parent task
  ID 和覆盖范围；不能把全部 5.0131% 的 family 时间当作该子任务的可优化份额。
- correctness 覆盖四个 rank slice、`valid_rows=0/1/47/48`、graph padding、非零脏
  buffer、residual 两输出及初始无 residual 分支。scatter 的 zero-fill 只有在证明所有
  目标元素被覆盖时才可删除；局部 batch/padding 路径必须仍然正确。
- composite case 按整段真实必要操作计时，作为这个完整链任务的唯一 scored case；
  component cases 标记 `unscored` 用于定位，防止重复计分。重复 replay 用固定输入
  与独立输出/正确重置状态，避免上次 residual/collective 输出污染下一次测量。

## 6. 执行流程和拟议命令

```text
check -> materialize -> prepare + baseline -> prepared
                                          -> run / resume
                                          -> collect + independent recheck
                                          -> e2e A/B
```

`needs_evidence` 在 check 阶段进入缺失输入清单；补证据后生成新的 resolved task/bundle
版本并重新检查。`not_recommended` 只出现在报告中。无需让 agent 先优化再发现没有输入。

以下是脚本实现后的使用示例，运行位置为新实验容器：

```bash
python3 /run/run_forge/check.py \
  --analysis-dir /shared_nfs/yaoc/work/infera-test/glm5.2-pd-optimization/workspace/kernel-analyze-poc/025/analysis \
  --kernelforge /KernelForge --config /work/forge-config.yaml

python3 /run/run_forge/prepare.py \
  --analysis-dir /shared_nfs/yaoc/work/infera-test/glm5.2-pd-optimization/workspace/kernel-analyze-poc/025/analysis \
  --task-id tp4_bf16_allreduce_001 \
  --output-dir /forge_workspace/tp4_bf16_allreduce_001 \
  --config /work/forge-config.yaml

python3 /run/run_forge/run.py \
  --bundle /forge_workspace/tp4_bf16_allreduce_001 --max-hours 3 --dry-run
python3 /run/run_forge/run.py \
  --bundle /forge_workspace/tp4_bf16_allreduce_001 --max-hours 3
python3 /run/run_forge/run.py \
  --bundle /forge_workspace/tp4_bf16_allreduce_001 --resume --max-hours 3

python3 /run/run_forge/collect.py \
  --bundle /forge_workspace/tp4_bf16_allreduce_001
```

`run.py` 最终生成的原生调用大致如下；路径、SNR 阈值取自已通过的 bundle，
用 subprocess argv 数组传递，不通过 shell 拼接配置内容：

```text
kernel-agents forge-loop
  --workspace <bundle>/repo
  --kernel <bundle>/repo/aiter/csrc/include/custom_all_reduce.cuh
  --driver <bundle>/repo/measurement/driver.py
  --program-md-file <bundle>/repo/measurement/program.md
  --invocation-spec-file <bundle>/repo/measurement/invocation_spec.json
  --task-type repository --framework aiter --operator-name custom_all_reduce
  --source-files <已解析的源码列表>
  --target-functions cross_device_reduce_2stage
  --fellow hip-fellow --gpu-target gfx950 --gpu-type mi355x
  --nproc-per-node 4 --bench-repeat 3 --snr-threshold <已校准阈值>
  --agent-backend codex --model gpt-5.6-sol --agent-reasoning-effort max
  --agent-fallback-provider none
  --no-prepare-task --profiling --max-hours 3 --no-experience-kb
  --experiments-dir <bundle>/artifacts/experiments
  --result-json <bundle>/artifacts/result.json
```

`--no-prepare-task` 只用于已通过独立 prepare 的 bundle；runner 启动前重查身份和
driver gate。无须让 Forge 再修改已固定的 oracle/case 契约。正式开始前保存 clean
baseline commit；Forge 仍会按自己的规则测量 pristine baseline 并做 KEEP 判定。

resume 必须回到同一 workspace，不能重新生成/覆盖 driver、program 或更改 ranks/cases。
GPU、镜像、source、reference、graph 模式等身份变化就创建新 campaign。runner 对
workspace 和同一设备集合加锁，首版顺序运行两个任务；超时/中断时清理自己启动的
process group 和 torchrun workers，保留原生恢复产物。

## 7. 结果验收与实施顺序

`collect.py` 要区分 `preparation_failed`、`interrupted`、`no_improvement`、
`microbench_improved`、`e2e_verified`；进程退出 0 不等于找到优化。
导出必须包含 pristine/best commit、按来源 repo 可应用的 diff、环境/source/build
身份、每 case 正确性与耗时、测量 spread、上游分数和外部 workload 权重、已覆盖与
未覆盖的算子边界。oracle/cases/fixtures 被改或 candidate 来源不明时不发布补丁。

Forge 当前 score 是 scored cases 的 `baseline_ms / candidate_ms` 等权算术平均，
并对候选做重复测量；报告不能将它直接表述为线上吞吐收益。正式 A/B 使用相同请求集、
seed、TP/DP/EP、EAGLE、graph 模式和补丁基础，先做至少三组交错的 baseline/candidate
无 profiler 对照，比较 output token/s、TPOT（含尾延迟）、accept length 和 output
tokens，并展示波动。`bench.sh` 当前每次准备 prompts，e2e 层需固定/核对请求输入；
最后短 trace 用于确认目标 dispatch 和算子链变化。

fake PD KV 和模拟 acceptance 的 benchmark 只支持这个合成 workload 的性能结论；
算子正确性由 driver 负责，真实 PD/模型输出正确性需另外验收。先单独接回两项补丁，
再测试组合版本，避免将 norm/collective 边界的收益重复归因。

建议按以下三个交付点实现：

1. **首个可运行任务包**：check/prepare、source snapshot、上述 ABI 修正、TP4 driver、
   graph/correctness/profile preflight、真实 baseline。CPU 验证 malformed readiness、
   renderer 不漏字段、case ID/路径映射、resume 身份拒绝等契约。
2. **一次完整 campaign**：run/resume/collect、进程和 GPU 锁、源码 rebuild 生效检查、
   独立 correctness/perf 复验，取得可应用补丁或明确 no-improvement 结果。
3. **扩大覆盖与 e2e**：norm/staging component + 四 rank chain，再实现 A/B 接回；
   同期补 MoE/attention/indexer 的动态输入。动态输入采样必须覆盖实际 replay 值，
   仅在 graph capture 时 hook Python 只能恢复构建信息。

本次已完成的检查是 YAML 重验、draft/task 一致性、family 数值核对、原容器源码/模型
参数核对和三个 KernelForge 接口文件的版本一致性检查。没有执行 kernel benchmark、
硬件 profiling、模型调用或 KernelForge 优化；性能收益与 driver 可运行性留给上述交付点验证。

## 参考位置

- [025 分析](../../workspace/kernel-analyze-poc/025/analysis/analysis.md)
- [all-reduce handoff](../../workspace/kernel-analyze-poc/025/analysis/tasks/tp4_bf16_allreduce_001.yaml)
- [norm/staging handoff](../../workspace/kernel-analyze-poc/025/analysis/tasks/residual_norm_quant_chain_001.yaml)
- [handoff 校验器](../analyze_profiles_by_agent/tools/validate_handoff.py)
- [KernelForge CLI](../../third-party/KernelForge/src/kernel_agents/cli.py)
- [task preparer](../../third-party/KernelForge/src/kernel_agents/loop/task_preparer.py)
- [program renderer](../../third-party/KernelForge/src/kernel_agents/loop/program.py)
- [all-reduce driver 示例](../../third-party/KernelForge/examples/aiter-allreduce-forge-loop/driver.py)
- [现有容器入口](../start_container.sh) / [benchmark](../run_profiles/bench.sh)
