#srun --pty --gres=gpu:1 --mem=4G --time=60 --partition=testgpu bash
module load cuda/10.1
srun --pty --account=erickson  --gres=gpu:1 --mem=32G --time=240 --partition=testgpu bash 
