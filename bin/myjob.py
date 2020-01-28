#!/usr/bin/env python
import subprocess
import os
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filter', type=str, default=None)
    parser.add_argument('-r', '--remove', type=int, default=0)
    parser.add_argument('-d', '--detail', type=int, default=0)
    parser.add_argument('-l', '--logs', type=int, default=0)
    args = parser.parse_args()

    output = subprocess.check_output(["kubectl", "get", "job"]).decode('utf-8').strip()
    jobs = []
    for i in output.split('\n'):
        if 'hza-job' not in i: 
            continue
        if args.filter is not None:
            if args.filter not in i:
                continue
        print(i)
        jobs.append(i.split(' ')[0])

    pods_id = [i.split(' ')[0] for i in subprocess.check_output(["kubectl", "get", "pods"]).decode('utf-8').strip().split('\n')]
    if args.logs:
        for i in jobs:
            for j in pods_id:
                if i == j[:-6]:
                    break
            os.system("kubectl logs {}".format(j))

    if args.detail:
        for i in jobs:
            os.system("kubectl describe job {}".format(i))
    
    if args.remove:
        print('Are you sure you want to delete the above files?(y/n)')
        y = input()
        if y == 'y':
            for j in jobs:
                os.system('kubectl delete job {}'.format(j))

if __name__ == '__main__':
    main()