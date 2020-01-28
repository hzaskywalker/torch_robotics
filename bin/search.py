#!/usr/bin/env python
"""
read a scripts
add several line to generate different scripts
create_name
"""
import os
import argparse
from mp.utils import mp_run
import hashlib, base64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('parameter', type=str)
    parser.add_argument('list', type=str)
    parser.add_argument('--type', type=str, default='float')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--job_name', type=str, default=None)
    args = parser.parse_args()

    from mp.bin.hza_try import create_hza_try
    create_hza_try()

    name = '--' + args.parameter

    cmd = open(args.file, 'r').readline().strip()
    a = cmd.split(' ')


    path = None
    for i in range(len(a)):
        if a[i] == '--path':
            path = a[i+1]
            a = a[:i] + a[i+2:]
            break

    assert path is not None or args.path is not None
    if args.path is not None:
        path = args.path


    for i in range(len(a)):
        if a[i] == name:
            a = a[:i] + a[i+2:]
            break
    a = ' '.join(a)
    print("BASED ON THE FOLLOWING Command:")
    print(a)

    new_cmds = []
    names = []
    for i in args.list.split(','):
        if args.type != 'str':
            i = eval('{}({})'.format(args.type, i))
        print(i, type(i))
        names.append('{}_{}'.format(args.parameter, i))

        new_cmds.append(a + ' --{} {}'.format(args.parameter, i) + ' --path {}'.format(os.path.join(path, names[-1])))

    print("We want to execute the following command:")
    for i in new_cmds:
        print(i)

    if args.run == 1:
        cmds = []
        for key, i in zip(names, new_cmds):
            #d=hashlib.md5(i.encode('ascii')).digest(); d=base64.b64encode(d); 
            #print(d.decode('ascii'))
            job_name = args.job_name
            if job_name is None:
                job_name = args.file.split('/')[-1].split('.')[0]

            file_name = os.path.join('/tmp', '{}_{}.sh'.format(job_name, key))
            with open(file_name, 'w') as f:
                f.write(i)
            print('write on file:', file_name)

            dir_path = os.path.dirname(os.path.abspath(__file__))
            job_name =  job_name + '-' + key.replace('.', '-').replace('_', '-')
            job_name = job_name.lower()
            cmd = 'cd {}/..;{}/run.py {} --job_name {} --copy 0'.format(dir_path, dir_path, file_name, job_name)
            print(cmd)
            cmds.append(cmd)

        print("Are you sure you want to run? (y/n)")
        y = input()
        if y == 'y':
            print('copy code...')
            os.system("kubectl cp /home/hza/mp hza-try:/cephfs/hza/")
            for cmd in cmds:
                os.system(cmd)

    elif args.run == 2:
        # test locally
        mp_run(new_cmds)
    else:
        print('just kidding, add --run 1 or --run 2 to run remotely or locally')

if __name__ == '__main__':
    main()