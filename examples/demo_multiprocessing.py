from tensorboardX import GlobalSummaryWriter
import multiprocessing as mp
import time
import os
import psutil
import torch
import numpy as np

w = GlobalSummaryWriter()


def train3():
    for i in range(100):
        w.add_scalar('many_write_in_func', np.random.randn())
        time.sleep(0.01*np.random.randint(0, 10))

def train2(x):
    np.random.seed(x)
    w.add_scalar('few_write_per_func/1', np.random.randn())
    time.sleep(0.05*np.random.randint(0, 10))
    w.add_scalar('few_write_per_func/2', np.random.randn())

def train(x):

    w.add_scalar('poolmap/1', x*np.random.randn())
    time.sleep(0.05*np.random.randint(0, 10))
    w.add_scalar('poolmap/2', x*np.random.randn())



if __name__ == '__main__':

    with mp.Pool() as pool:
        pool.map(train, range(100))


    processes = []
    for i in range(4):
        p0 = mp.Process(target=train2, args=(i,))
        p1 = mp.Process(target=train3)
        processes.append(p0)
        processes.append(p1)
        p0.start()
        p1.start()

    for p in processes:
        p.join()

    w.close()