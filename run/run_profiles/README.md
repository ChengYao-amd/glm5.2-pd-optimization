# GLM-5.2 fake-decode profile

在 `Dockerfile.kernelforge` 构建的容器内使用。默认配置集中在
`/run/env.bashrc`：模型 `/shared_nfs/models/GLM-5.2-MXFP4`，GPU 0–3，
TP4/DP4/EP1（DP attention 开启，无 EP），FP8 KV，mem-fraction 0.85，并发 32，EAGLE 5/6/topk1，
模拟接受长度 3.61，HiCache 关闭。

```bash
# 宿主机：创建容器并进入，挂载 /shared_nfs 和 run
bash run/start_container.sh
docker exec -it dev-container bash

# 容器终端 1：前台启动，Ctrl-C 停止
bash /run/run_profiles/up.sh

# 容器终端 2：等 server ready 后运行
bash /run/run_profiles/bench.sh
bash /run/run_profiles/profile.sh

# 临时覆盖配置；并发/并行参数变化后需以相同配置重启 server
OUT_DIR=/run/run_profiles/results/test1 PROFILE_STEPS=24 bash /run/run_profiles/profile.sh
```

三个脚本自动加载 `env.bashrc`，末尾参数原样传给 SGLang。
`bench.sh` 使用本地生成的原版 synthetic prompts：16 个 warmup（输出 32 tokens），
128 个测量请求，每个输入 10000 / 输出 500 tokens。`profile.sh` 在 warmup 后
自动采集 12 个 forward 并停止；采集从测量请求开始，包含批次爬升阶段，不等待满并发。
结果默认写入 `results/bench-时间/` 或 `results/profile-时间/`，含 `benchmark.jsonl`；
trace 位于 `traces/<时间>/*.trace.json.gz`，按 TP rank 分文件，可用 Perfetto 打开。
`OUT_DIR` 指定 profile 输出时须使用新目录。模型通过完整的 `/shared_nfs` 挂载访问，
以保留 Hugging Face snapshot 到 `blobs` 的相对软链接；外部模型路径需自行增加挂载。

`up.sh` 自动向容器内 `/sglang` 应用 `fake_decode.patch`，重复启动会跳过已应用的补丁。
补丁来自 `Infera-glm-5.2-exp` 的
`packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810/patches/`，
对应 SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`，与当前 Dockerfile 一致。

默认 `DEBUG_CLR_GRAPH_PACKET_CAPTURE=false`，保留 CUDA Graph，同时让 ROCm 7.2
图内 kernel 出现在 trace 中。需要原模式性能时，用
`DEBUG_CLR_GRAPH_PACKET_CAPTURE=true bash /run/run_profiles/up.sh` 重启后运行 `bench.sh`；
两种模式的图回放开销不同。这里的 fake KV 和模拟接受只用于 synthetic decode 测量。
