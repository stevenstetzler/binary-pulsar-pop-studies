import numpy as np
import multiprocessing as mp

def process_wrapper(lineByte):
    with open("binary/pars_H3_H4/B1953+29/chains/chain_1.txt") as f:
        f.seek(lineByte)
        line = f.readline()
       	data = np.fromstring(line)
	print data[0]

#init objects
pool = mp.Pool(8)
jobs = []

#create jobs
with open("binary/pars_H3_H4/B1953+29/chains/chain_1.txt") as f:
    nextLineByte = f.tell()
    for line in f:
        jobs.append( pool.apply_async(process_wrapper,(nextLineByte)) )
        nextLineByte = f.tell()

#wait for all jobs to finish
for job in jobs:
    job.get()

#clean up
pool.close()
