# hold a node

ACCOUNT="amd-oai"
# ACCOUNT="amd-agentx-2"

QOS="amd-oai-qos"
# QOS="amd-agentx-2-qos"

PARTITION="amd-spur"

sbatch -A $ACCOUNT -p $PARTITION --gres=gpu:8 --qos=$QOS --output=stdout.log --nodes=1 --error=stderr.log -t 24:00:00 --job-name=hold run/cluster_utils/sleep.sh

# Use the following command to run a task on the held node:
# spur exec <job_id> bash -c "<command>"


# build image
# spur exec 133995 bash -c "cd /shared_nfs/yaoc/work/infera-test/rocm-llm-bench && docker build -f Dockerfile.kernelforge -t rocm-llm-bench:kernelforge ."