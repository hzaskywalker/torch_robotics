#!/usr/bin/env python3
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, default=None)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--job_name', type=str, default=None)
    parser.add_argument('--args', type=str, default=None)
    parser.add_argument('--copy', type=int, default=1)
    args = parser.parse_args()

    if args.run:
        print('copy code and command...')
        if args.copy:
            os.system("kubectl cp /home/hza/Robotics hza-try:/cephfs/hza/")
        os.system("kubectl cp {} hza-try:/cephfs/hza/tmp/".format(args.file))

    basename = os.path.basename(args.file)
    if args.job_name is None:
        jobname = basename.split('.')[0]
    else:
        jobname = args.job_name

    lines = []
    template_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')
    with open(os.path.join(template_loc, 'job_template.yaml'), 'r') as f:
        for i in f.readlines():
            if 'args' in i:
                lines.append(i.replace('/cephfs/hza/mp/mp/scripts/cem.sh', 'bash /cephfs/hza/tmp/{}'.format(basename)))
            elif "  name:" in i:
                lines.append(i.replace('hza-try-job', 'hza-job-{}'.format(jobname)))
            else:
                lines.append(i)

    yaml_file = '/tmp/{}.yaml'.format(jobname) 
    with open(yaml_file, 'w') as f:
        for i in lines:
            if not args.run:
                print(i, end='')
            f.write(i)

    if args.run:
        os.system("kubectl create -f {}".format(yaml_file))

if __name__ == '__main__':
    main()