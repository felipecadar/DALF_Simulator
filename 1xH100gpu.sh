srun --ntasks=1 --cpus-per-task=40 --gres=gpu:1 --time=00:30:00 -C h100 --partition=gpu_p6 -A xab@h100 --pty bash -i
