chmod +x *.py

rm -f log*

export CUDA_VISIBLE_DEVICES=-1
nohup python3 main.py > logps --job_name=ps --task_index=0 2>&1 &

export CUDA_VISIBLE_DEVICES=1
nohup python3 main.py > logworker0 --job_name=worker --task_index=0 2>&1 &

export CUDA_VISIBLE_DEVICES=2
nohup python3 main.py > logworker1 --job_name=worker --task_index=1 2>&1 &

tailf logps

