#!/usr/bin/env python
# monitor, file edit, and test
import os

def create_hza_try():
    template_loc = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')
    os.system("kubectl create -f {}/try.yaml".format(template_loc))


def main():
    create_hza_try()
    os.system("kubectl exec hza-try -it -- /bin/bash")

if __name__ == '__main__':
    main()