import sapien.core as sapien_core
from multiprocessing import Process

def run():
    sim = sapien_core.Simulation()
    return sim

if __name__ == '__main__':
    workers = []
    sim = run()
    for i in range(5):
        workers.append(Process(target=run))
    for i in workers:
        i.start()
    for i in workers:
        i.join()
