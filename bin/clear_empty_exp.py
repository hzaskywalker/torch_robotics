#!/usr/bin/env python
import sys
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('loc', type=str)
    args = parser.parse_args()

    for i in os.listdir(args.loc):
        t = os.path.join(args.loc, i)
        f = open(os.path.join(t, 'log'), 'r').readlines()
        if len(f) < 5:
            print(t, f, "are you sure you want to delete") 
            a = input()
            if a=='y':
                os.system('rm -rf {}'.format(t))

if __name__ == '__main__':
    main()