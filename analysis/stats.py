from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from analysis_utils import *
import sys


# Return the emipircal CDF of a set of data
def get_cdf(data):
	np_data = np.array(data)
	data_sorted = data
#	data_sorted = np.sort(data)
	return data_sorted.cumsum() / data_sorted.sum()


def stats_vals():
	data = np.genfromtxt("/users/sstetzle/nanograv_data/plots/par_estimates/COSI_values.txt")
	cosi = data[:, 1]
	cosi_cdf = get_cdf(cosi)
	sorted_cosi_cdf = get_cdf(sorted(cosi))

	plt.hist(cosi)
	plt.show()

	n = 1000
	uniform_data = stats.uniform.pdf(np.linspace(0, 1, n))
	uniform_cdf = get_cdf(uniform_data)

	ks_stat, p_val = stats.kstest(cosi, 'uniform')
	plt.plot(np.linspace(0, 1, len(cosi)), cosi_cdf, label='cosi            ks={0:.2f} p={1:.2f}'.format(ks_stat, p_val))

	ks_stat, p_val = stats.kstest(sorted(cosi), 'uniform')
	plt.plot(np.linspace(0, 1, len(cosi)), sorted_cosi_cdf, label='sorted cosi ks={0:.2f} p={1:.2f}'.format(ks_stat, p_val))

	ks_stat, p_val = stats.kstest(uniform_data, 'uniform')
	plt.plot(np.linspace(0, 1, n), uniform_cdf, label='uniform      ks={0:.2f} p={1:.2f}'.format(ks_stat, p_val))
	plt.xlabel("COSI")

	plt.legend()
	plt.savefig("/users/sstetzle/data_uniform_cdf.png")
	plt.show()


def stats_dists(pulsar_dirs):
	dicts = []
	chain = None
	for pulsar_dir in pulsar_dirs:
		pulsar_dict = get_pulsar_dict(pulsar_dir)
		for pulsar in pulsar_dict:
			_, _, par_dict = pulsar_dict[pulsar]
			if chain is not None:
				chain = np.concatenate((chain, par_dict['COSI']))
			else:
				chain = par_dict['COSI']

	plt.plot(chain)
	plt.show()

def main():
	pulsar_dirs = sys.argv[1:len(sys.argv)]
	stats_dists(pulsar_dirs)

if __name__ == '__main__':
	main()
