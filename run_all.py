import subprocess
import numpy as np
import time
import os
from tqdm import tqdm

def main():
    processes = []
    commands = []
    for index in [1, 2, 3]:
	    for data in ['GAS', 'POWER', 'HEPMASS', 'MINIBOONE', 'BSDS300']:
	    	for model in ['RealNVP', 'MAF', 'GLOW', 'SPLINE-AR']:
    			commands.append(f"sbatch --gpus=1 -p normal -c 4 run_nfcalib.sh --data {data} --model {model} --index {index}")

    batch_size = 20
    for command in tqdm(commands):
        print(command)
        process = subprocess.Popen(command,
                                   shell=True,
                                   close_fds=True,
                                   # stdout=subprocess.DEVNULL,
                                   # stderr=subprocess.DEVNULL
                                   )
        processes.append(process)
        pr_count = subprocess.Popen("squeue | grep martemev | wc -l", shell=True, stdout=subprocess.PIPE)
        out, err = pr_count.communicate()
        if int(out) > batch_size:
            while int(out) > batch_size:
                print("Waiting... ")
                time.sleep(240)
                pr_count = subprocess.Popen("squeue | grep martemev | wc -l", shell=True, stdout=subprocess.PIPE)
                out, err = pr_count.communicate()

    for process in processes:
        print(process.pid)


if __name__ == "__main__":
    main()

