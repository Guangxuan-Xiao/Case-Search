from calendar import c
import subprocess
import argparse
import os.path as osp
import os
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, required=True)
    parser.add_argument('--max-jobs', type=int, required=True)
    parser.add_argument('--cuda-devices', type=str,
                        required=True)  # e.g. '0,1,2,3'
    parser.add_argument('--sleep', type=int, default=1)
    return parser.parse_args()


def main(args):
    cuda_devices = args.cuda_devices.split(',')
    configs = [osp.join(args.config_dir, f)
               for f in os.listdir(args.config_dir) if f.endswith('.yml')]
    config_nums = len(configs)
    print(f'Found {config_nums} configs')
    for i in range(0, config_nums, args.max_jobs):
        processes = []
        for j in range(args.max_jobs):
            if i + j >= config_nums:
                break
            print(f'Running job {i + j}/{config_nums}')
            config = configs[i + j]
            cuda_device = cuda_devices[j]
            cmd = f'CUDA_VISIBLE_DEVICES={cuda_device} python src/main.py --config {config}'
            processes.append(subprocess.Popen(cmd, shell=True))
            time.sleep(args.sleep)
        for p in processes:
            p.wait()
        for j in range(args.max_jobs):
            if i + j >= config_nums:
                break
            config = configs[i + j]
            os.rename(config, f'{config}_done')


if __name__ == '__main__':
    args = parse_args()
    main(args)
