import cPickle as pickle
import numpy as np
import argparse
from scipy.stats import kstest
from scipy.stats import ks_2samp as kstest2
from matplotlib import pyplot as plt

def sample(chains):
	ret_samples = np.zeros(len(chains))
	for chain, i in zip(chains, np.arange(0, len(chains))):
		index = np.random.randint(0, len(chain) - 1)
		ret_samples[i] = chain[index]
	return ret_samples

def run_simulation_ks_one_samp(chains, niter):
	d_vals = np.zeros(niter)
	p_vals = np.zeros(niter)

	chains_0 = []

	for i in range(niter):
		curr_sample = sample(chains)
#		print curr_sample
		chains_0.append(curr_sample[0])
		d, p = kstest(curr_sample, 'uniform')
#		d, p = kstest(curr_sample, lambda x : x)
		#print "P-Val: {}".format(p)
		d_vals[i] = d
		p_vals[i] = p

	ax1 = plt.subplot(211)
	ax1.hist(p_vals)
	ax1.set_title("p vals")
	ax2 = plt.subplot(212)
	ax2.hist(d_vals)
	ax2.set_title("d vals")
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

	plt.show()

#	ax1 = plt.subplot(211)
#	ax1.hist(chains[0], bins=80, range=(0, 1))
#	ax1.set_title("Originial Chain")
#	ax2 = plt.subplot(212)
#	ax2.hist(chains_0, bins=80, range=(0, 1))#, histtype='step')
#	ax2.set_title("Sampled Chain")
#        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
#
#	plt.show()

	p_10 = [p for p in p_vals if p < 0.1]
	return float(len(p_10))/niter


def run_simulation_ks_two_samp(chains, niter):
	d_vals = np.zeros(niter)
	p_vals = np.zeros(niter)
	
	chains_0 = []

	compare_sample_size = 100000
	prev_sample = np.random.uniform(0, 1, compare_sample_size)
	for i in range(niter):
		curr_sample = sample(chains)
		#print curr_sample
		chains_0.append(curr_sample[0])
		compare_sample = np.random.uniform(0, 1, compare_sample_size)
		while prev_sample is compare_sample:
			compare_sample = np.random.uniform(0, 1, compare_sample_size)
		prev_sample = compare_sample	
#		print compare_sample
		d, p = kstest2(curr_sample, compare_sample)	
#		d, p = kstest2(curr_sample, lambda x : x)
#		print "P-Val: {}".format(p)
		d_vals[i] = d
		p_vals[i] = p

	ax1 = plt.subplot(211)
	ax1.hist(p_vals)
	ax1.set_title("p vals")
	ax2 = plt.subplot(212)
	ax2.hist(d_vals)
	ax2.set_title("d vals")
        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)

	plt.show()

#	ax1 = plt.subplot(211)
#	ax1.hist(chains[0], bins=80, range=(0, 1))
#	ax1.set_title("Originial Chain")
#	ax2 = plt.subplot(212)
#	ax2.hist(chains_0, bins=80, range=(0, 1))#, histtype='step')
#	ax2.set_title("Sampled Chain")
#        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
#
#	plt.show()

	p_10 = [p for p in p_vals if p < 0.1]
	return float(len(p_10))/niter


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--chains_file", help="The pickle file of cosi chains to sample over.")
	parser.add_argument("--niter", type=int, help="The number of iterations to run the simulation for.", required=True)
	
	args = parser.parse_args()
	
	chains_file = args.chains_file
	niter = args.niter	

	try:
		cosi_chains, _ = pickle.load(open(chains_file, "rb"))
	except:
		cosi_chains = pickle.load(open(chains_file, "rb"))
	
	one_sample_reject = run_simulation_ks_one_samp(cosi_chains, niter)
	print "One Sample KS Test: {}% with p below 10%".format(100.*one_sample_reject)

	two_sample_reject = run_simulation_ks_two_samp(cosi_chains, niter)
	print "Two Sample KS Test: {}% with p below 10%".format(100.*two_sample_reject)


if __name__ == '__main__':
	main()

