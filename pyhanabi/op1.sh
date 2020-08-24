#!/bin/bash -l
#SBATCH --job-name=op1
#SBATCH --account=project_2003228
#SBATCH --time=10:00:00
#SBATCH --mem=40G
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

source /projappl/project_2003228/miniconda3/etc/profile.d/conda.sh
activate_hanabi
srun python main.py \
       --save_dir exps/op1 \
       --other_play True \
       --load_index 1 \
       --num_thread 40 \
       --num_game_per_thread 40 \
       --method vdn \
       --greedy_extra 1 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 1e-04 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 9 \
       --batchsize 64 \
       --burn_in_frames 10000 \
       --replay_buffer_size 31072 \
       --epoch_len 400 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --act_device cpu

