# hold a node

ACCOUNT="amd-oai"
# ACCOUNT="amd-agentx-3"
# ACCOUNT="amd-primus"

QOS="amd-oai-qos"
# QOS="amd-agentx-3-qos"
# QOS="amd-primus-qos"

PARTITION="amd-spur"

sbatch -A $ACCOUNT -p $PARTITION --gres=gpu:8 --qos=$QOS --output=stdout.log --nodes=1 --error=stderr.log -t 24:00:00 --job-name=hold run/cluster_utils/sleep.sh

# Use the following command to run a task on the held node:
# spur exec <job_id> bash -c "<command>"
# or like this: srun --jobid=<job_id> --nodelist=<node_name> --overlap bash -c 'docker ps'
