#srun --pty --gres=gpu:1 --mem=4G --time=60 --partition=testgpu bash
srun --pty --account=erickson  --gres=gpu:1 --mem=8G --time=240 --partition=testgpu bash 
