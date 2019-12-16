import os
from multiprocessing import Process

def work(args):
    os.system(args)

def mp_run(codes, *args, func=None):
    if isinstance(codes, list):
        pass
    elif isinstance(codes, str):
        codes = [codes] + list(args)
    else:
        raise NotImplementedError

    # run multi training code at the same time
    tasks = []
    if func is None:
        func = work
    for c in codes:
        t = Process(target=work, args=(c,))
        tasks.append(t)
    for t in tasks:
        t.daemon = True
        t.start()
    for t in tasks:
        t.join()