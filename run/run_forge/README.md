# KernelForge task runner

第一版提供 `check.py`、`prepare.py`、`run.py`、`collect.py`，以及三个经过 MI355X
验证的任务模板。原分析与完整方案保存在 [DESIGN.md](DESIGN.md)。当前实现运行
独立算子/局部链；serving 接回和 e2e A/B 尚未实现，不输出 e2e 收益结论。

| task ID | 来源及范围 | GPU |
|---|---|---:|
| `tp4_bf16_allreduce_001` | 025同名ready任务；BF16 `[192,6144]` raw out-of-place all-reduce，三种callsite输入 | 4 |
| `dp_staging_001` | 025 norm/staging子任务；原fill+copy，79 gather + 78 scatter局部composite | 1 |
| `residual_rmsnorm_001` | 025 norm/staging子任务；BF16 `[48,6144]` residual add + RMSNorm，双输出 | 1 |
| `mxfp4_moe_experts_001` | 实采TP0第3/39/77层；MXFP4排序、量化、两级expert GEMM及路由累加 | 1 |

MoE已在9月15日补齐本轮试验的实际输入；attention/indexer仍需动态输入证据。子任务不继承整个family的性能份额。
025稳定窗口中all-reduce约11.334%、本次staging约3.377%、带residual的norm约1.626%，
均为跨rank kernel duration比例，不能视为e2e比例。

## 环境和文件

在专用ROCm容器中执行，需MI355X/gfx950、PyTorch/ROCm/Triton、KernelForge。
AITER任务还需相应源码和构建依赖。沿用`run/start_container.sh`；初始源码为
`/aiter`、`/sglang`，KernelForge在`/KernelForge`。

```bash
python3 -m pip install -r /run/run_forge/requirements.txt
python3 /run/run_forge/apply_compat.py --kernelforge /KernelForge
```

适配器校验KernelForge `cd9c5850699b0550c2aa06be83c3645cf4e98e24`的三个关键接口hash。
任务recipes在`lib/common.py`，driver起点及GPU测试证据在`templates/`。
`apply_compat.py`只用于专用容器：修复该版本Codex guard误拒绝Analysis产物的问题，
只允许显式声明的分析输出目录，保留source/driver/Git保护。补丁和17项保护回归测试在
`patches/`，安装后的文件hash写入manifest；原始第三方checkout保持不变。
每任务一个bundle，正式实验位于`workspace/forge-workspace/<node>/<task-id>`：

```text
input/                 原task、修正后task、证据与来源manifest
manifest.json          源码/镜像/runtime身份、测量文件hash、参数
repo/                  独立Git仓库，forge-<task-id>开发分支
  aiter/               需要时复制tracked源码及submodules，不共享.git
  measurement/         kernel、driver、reference、harness、cases、program/spec
  forge_experiments/   原生campaign_config/run_state/checkpoints
artifacts/             preflight、baseline、experiments、日志和result
run.json               累计预算、session/PID/heartbeat/退出状态
state/codex/           持久SDK会话，供后续恢复
export/                报告、独立复验和通过后导出的补丁
```

standalone任务仅修改`measurement/kernel.py`及`candidate_`前缀helper；all-reduce
修改隔离AITER树的csrc/dist。driver/reference/harness/cases/program/spec固定并校验hash。
原安装目录与分析产物不参与优化修改。

runtime显式加载当前bundle源码，使用独立cache。初测cache在节点本地`/tmp`，
Forge随后可能把AITER cache重定位到bundle的experiments目录；ROCm TMPDIR仍在本地。
`ROCPROF_TMPDIR`另设在私有本地目录，防止profiler把临时counter文件写入源码目录。
正式环境设置`DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`。未设置该变量时的早期模板测试
数据只能证明调通，不能与正式baseline直接比较。

## 使用

命令在目标容器内运行，BUNDLE需为新目录：

```bash
ANALYSIS=/shared_nfs/yaoc/work/infera-test/glm5.2-pd-optimization/workspace/kernel-analyze-poc/025/analysis
BUNDLE=/shared_nfs/yaoc/work/infera-test/glm5.2-pd-optimization/workspace/forge-workspace/025/tp4_bf16_allreduce_001

python3 /run/run_forge/check.py --analysis-dir "$ANALYSIS" --kernelforge /KernelForge
python3 /run/run_forge/prepare.py --analysis-dir "$ANALYSIS" \
  --task-id tp4_bf16_allreduce_001 --output-dir "$BUNDLE" --node 025
python3 /run/run_forge/run.py --bundle "$BUNDLE" --max-hours 12 --dry-run
python3 /run/run_forge/run.py --bundle "$BUNDLE" --max-hours 12
```

prepare用完整模板执行上游preflight（correctness、bench、真实graph、profile），
随后三次独立benchmark，每次`--repeat 3`。通过后才标`prepared`。
`--materialize-only`只建包，之后用`--verify --output-dir ...`完成验证。
`--config`提供默认参数，显式CLI参数优先；示例见`config.example.yaml`。

MoE使用`--fixture-dir`和实采handoff。先`--materialize-only`，执行bundle中
`measurement/prepare_goldens.py`复现采样输出、生成原实现goldens并校准数值门槛，
再固定cases/goldens/hash后进行`--verify`。完整实跑步骤和记录见
`.record/20260915-moe-forge/`；不要把尚未校准的MoE模板直接提交优化。

需要agent修复driver时增加`--repair-driver`，调用上游`prepare_task_sync`；
reference/cases等仍受保护。当前三个模板不需要agent修复。
`check.py --driver-bundle "$BUNDLE"`可独立重新检查driver。

Agent配置复用mounted `/llm_gateway/config.toml`及`credentials.json`，进程内转换为
Forge需要的OPENAI gateway环境。凭据不写入本项目参数/manifest，launcher输出脱敏。
provider/model固定为`codex`/`gpt-5.6-sol`，effort `max`被当前Forge映射为`xhigh`。
自动跨provider fallback与外部experience KB关闭。

## 预算、后台运行和恢复

`--max-hours 12`表示同一bundle**累计12小时**。恢复仍传12，runner扣除已用session时间。
`--allocation-deadline-unix <epoch>`使Forge在allocation结束前5分钟收尾，另有
supervisor watchdog和实验启动脚本的外部timeout兜底。

```bash
python3 /run/run_forge/run.py --bundle "$BUNDLE" --resume --max-hours 12 \
  --allocation-deadline-unix <new-allocation-end-epoch>
```

保持同一bundle路径、源码、driver/cases、设备布局和软件runtime身份。换allocation时
仍需挂载原路径及匹配运行环境。极少量剩余预算不足原生CLI最小1小时的情况，绝对deadline
仍限制实际运行时间。prepare/qualification时间不计入12小时优化预算。

启动失败且尚无原生checkpoint时用`--retry-start`；要求仍为pristine commit，保留失败
记录并累计已用时间。正常恢复用`--resume`。运行中的状态看`run.json`与原生run_state，
旧session退出文件不是当前运行状态。

本次后台启动及allocation绑定见`.record/20260914-forge-poc/launch-node.sh`。
通过`docker exec -d`脱离交互终端，自己的GPU进程在allocation结束前停止。

## 查看与导出

本次三节点实验可从项目根目录执行
`python3 .record/20260914-forge-poc/status.py --watch --interval 30`。
完整追踪说明见 [TRACKING.md](../../.record/20260914-forge-poc/TRACKING.md)。

```bash
python3 /run/run_forge/collect.py --bundle "$BUNDLE" --report-only
python3 /run/run_forge/collect.py --bundle "$BUNDLE"
```

`--report-only`不使用GPU，可汇总运行状态。默认collect需任务已停止，重新验证
correctness并做三次性能测量；测量文件/修改范围完整、确有改进时才导出补丁。
all-reduce导出相对AITER根的`aiter.patch`；standalone任务导出
`standalone-kernel.patch`，需要另行集成callsite及serving A/B。

报告保留等权case speedup及按源码调用次数加权的component时间估计。
GPU-event graph replay latency不等于线上critical-path时间；RMSNorm采用128次同buffer
调用/graph摊薄host提交开销，真实working set及边界依赖需集成后验证。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s run/run_forge -p test_contracts.py
```

CPU测试覆盖case完整性、reference改动拒绝、累计预算、argv/resume语义、启动所有者审计、
kept revision选择和跨bundle设备锁。所有GPU入口共用设备锁；`--node`应使用物理节点标识。
GPU证据在template说明和bundle的`artifacts/preflight.json`、`baseline.json`中。
错误算子/空graph负例验证过driver失败时非零退出；进程退出0本身不代表找到优化。

TP/DP 或输入 shape 与内置模板不同的实验，可传 `--recipe-file recipe.yaml`
和 `--template-dir /path/to/measurement-template`。Recipe 使用 `lib/common.py`
内置 recipe 的字段，并必须提供实际 `workload`；`nproc` 决定默认设备列表。
完整 driver/reference/cases/harness 仍会复制、冻结并通过同一套 GPU preflight 与
baseline 校验。实验 recipe 会归档到 bundle 的 `input/recipe.yaml`。不要复用旧
模板的 shape、校准记录或 TP4 性能数据作为新 workload 的证据。
