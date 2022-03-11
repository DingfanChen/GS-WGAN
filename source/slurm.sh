#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:2

__conda_setup="$('/BS/dchen-projects/work/Software/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/BS/dchen-projects/work/Software/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/BS/dchen-projects/work/Software/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/BS/dchen-projects/work/Software/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate gswgan-pytorch

python main.py -data 'mnist' -name 'ResNet_slurm_2GPUs' -ngpus 2 -ldir '../results/mnist/pretrain/ResNet_default'