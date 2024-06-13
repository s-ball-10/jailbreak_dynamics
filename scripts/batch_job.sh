#!/bin/bash

cd
cd jailbreak_dynamics
source .venv/bin/activate

# Install yq if it is not already installed
if ! command -v yq &> /dev/null
then
    echo "yq not found. Installing..."
    wget https://github.com/mikefarah/yq/releases/download/v4.6.3/yq_linux_amd64 -O ./.venv/bin/yq
    chmod +x ./.venv/bin/yq
fi

# Load the config file
config_file="configs/config_SLURM_jobs.yaml"

# Extract the values of the variables in the config using yq
## Type
time=$(yq e '.args.time' $config_file)
ntasks=$(yq e '.args.ntasks' $config_file)
job=$(yq e '.args.job' $config_file)
partition=$(yq e '.args.partition' $config_file)
gres=$(yq e '.args.gres' $config_file)
export=$(yq e '.args.job' $config_file)

if [[ $job == 'salloc' ]]; then
    # Actions to be performed when $job equals 'salloc'
    echo "salloc"
    salloc --partition=$partition --gres=$gres --time=$time --nodes=$nodes --ntasks=$ntasks --mem-per-gpu=$mem_per_gpu --ntasks-per-node=$ntasks_per_node --cpus-per-task=$cpus_per_task
    
else
    # Actions to be performed when $job is not equal to 'salloc'
    echo "sbatch"
    sbatch --partition=$partition --gres=$gres --time=$time --ntasks=$ntasks --job-name=$job_name $job 
    
fi

squeue