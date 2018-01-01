import numpy as np
from itertools import islice
from psr_constants import Tsun
from psr_constants import SECPERDAY


par_file = "binary/pars_H3_H4/B1953+29/chains/pars.txt"
chain_file = "binary/pars_H3_H4/B1953+29/chains/chain_1.txt"

pars = np.genfromtxt(par_file, dtype=str)
h3_idx = np.where(pars == 'H3')[0][0]
h4_idx = np.where(pars == 'H4')[0][0]

N = 100

x_edge = np.linspace(0, 1, 100)

H, edges = np.histogram([], bins=x_edge)

with open(chain_file, 'r') as infile:
	i = 1
	while True:
		print "{}           \r".format(N * i),
		i += 1
		gen = islice(infile, N)
		arr = np.genfromtxt(gen, dtype=None)
		h3 = arr[:, h3_idx]
		h4 = arr[:, h4_idx]

		stig = np.divide(h4, h3)
		sini = 2. * stig / (1. + np.power(stig, 2.))
		cosi = np.sqrt(1. - np.power(sini, 2.))
#		print cosi

		H += np.histogram(cosi, bins=x_edge)[0]
		if arr.shape[0] < N:
			break	

print H

