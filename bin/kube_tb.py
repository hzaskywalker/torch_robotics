#!/usr/bin/env python
import os
import torch
import subprocess
import argparse

def main():
    parsre = argparse.ArgumentParser()
    args = parsre.parse_args()

    os.system("kubectl create -f template/kube_tb.yaml")
    out = subprocess.check_output(['bash', '-c', 'kubectl get pod |grep hza-tb']).decode('utf-8').strip()
    os.system("kubectl port-forward {} 6006".format(out.split(' ')[0]))


if __name__ == '__main__':
    main()