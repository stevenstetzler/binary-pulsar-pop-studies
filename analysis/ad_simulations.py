#!/users/sstetzle/bin/python
import cPickle as pickle
import numpy as np
import argparse
from scipy.stats import kstest
from scipy.stats import ks_2samp as kstest2
from scipy.stats import anderson_ksamp as ksamp
from matplotlib import pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def sample(chains, scaling=1):
	if type(scaling) is not int:
		raise Exception("Scaling passed to sample must be an integer")

	ret_samples = np.zeros(scaling * len(chains))

	for j in range(0, scaling):
		for chain, i in zip(chains, np.arange(0, len(chains))):
			index = np.random.randint(0, len(chain) - 1)
			ret_samples[i + j*len(chains)] = chain[index]
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


def get_sample_from_distribution(sample_size, distribution_type, **kwargs):
	if distribution_type == 'uniform':
		if not 'min' in kwargs.keys():
			raise Exception("Must pass min=min_val to 'get_sample_from_distribution' if uniform distribution specified.")
		if not 'max' in kwargs.keys():
			raise Exception("Must pass max=max_val to 'get_sample_from_distribution' if uniform distribution specified.")
		min_val = kwargs['min']
		max_val = kwargs['max']
		return np.random.uniform(min_val, max_val, sample_size)
	elif distribution_type == 'normal':
		if not 'mean' in kwargs.keys():
			raise Exception("Must pass mean=mean_val to 'get_sample_from_distribution' if normal distribution specified.")
		if not 'scale' in kwargs.keys():
			raise Exception("Must pass scale=scale_val to 'get_sample_from_distribution' if normal distribution specified.")
		mean = float(kwargs['mean'])
		scale = float(kwargs['scale'])
		return np.random.normal(loc=mean, scale=scale, size=sample_size)
	else:
		print "Distribution type {} is not implemented for 'get_sample_from_distribution'".format(distribution_type)
		return None


def run_simulation_ad_ksamp(chains, par_range, distribution_type='uniform', niter=10000, chain_samples_scaling=1, compare_sample_size=None, save_name=None, verify_sampling=False, visualize=False, **kwargs):
	if distribution_type == 'normal' and not 'mean' in kwargs.keys() and not 'scale' in kwargs.keys():
		raise Exception("Must pass mean=mean_val and scale=scale_val to 'run_simulation_ad_ksamp' if simulating with normal distribution")
	if compare_sample_size is None:
		compare_sample_size = len(chains) * chain_samples_scaling

	print "Sampling {} chains to determine Anderson-Darling statistic distribution.".format(len(chains))
	print "Sampling each chain {} times for {} iterations and comparing to {} samples.".format(chain_samples_scaling, niter, compare_sample_size)
	d_vals = np.zeros(niter)
	p_vals = np.zeros(niter)
	
	chains_0 = np.zeros(niter)

	if distribution_type == 'uniform':
		min_par, max_par = par_range
		prev_sample = get_sample_from_distribution(compare_sample_size, distribution_type, min=min_par, max=max_par)
	elif distribution_type == 'normal':
		mean_val = kwargs['mean']
		scale_val = kwargs['scale']
		prev_sample = get_sample_from_distribution(compare_sample_size, distribution_type, mean=mean_val, scale=scale_val)
	else:
		print "Distribution type {} not supported for anderson darling sampling.".format(distribution_type)
		return None

	for i in range(niter):
		p = 1.1
		while p > 1:
			curr_sample = sample(chains, scaling=chain_samples_scaling)
			chains_0[i] = curr_sample[0]
			if distribution_type == 'uniform':
				compare_sample = get_sample_from_distribution(compare_sample_size, distribution_type, min=min_par, max=max_par)
			elif distribution_type == 'normal':
				compare_sample = get_sample_from_distribution(compare_sample_size, distribution_type, mean=mean_val, scale=scale_val)
			while prev_sample is compare_sample:
				if distribution_type == 'uniform':
					compare_sample = get_sample_from_distribution(compare_sample_size, distribution_type, min=min_par, max=max_par)
				elif distribution_type == 'normal':
					compare_sample = get_sample_from_distribution(compare_sample_size, distribution_type, mean=mean_val, scale=scale_val)
			prev_sample = compare_sample
			d, _, p = ksamp([curr_sample, compare_sample])
		if visualize or save_name is not None and i % int(niter / 10) == 0:
			ecdf_sample = ECDF(curr_sample)
			ecdf_compare = ECDF(compare_sample)
			x_sample = np.linspace(min(curr_sample), max(curr_sample), len(curr_sample))
			x_compare = np.linspace(min(compare_sample), max(compare_sample), len(compare_sample))
			plt.plot(x_sample, ecdf_sample(x_sample), label="Sample")
			plt.plot(x_compare, ecdf_compare(x_compare), label="Compare")
			plt.legend()
			if save_name is not None:
				sampling_save_name = save_name.split(".png")[0] + "_ecdf_check_{}.png".format(i / int(niter / 10))
				print "Saving figure to {}".format(sampling_save_name)
				plt.savefig(sampling_save_name)
			if visualize:
				plt.show()
			plt.close('all')
		d_vals[i] = d
		p_vals[i] = p

	p_10 = [p for p in p_vals if p < 0.1]
	p_10_ratio = float(len(p_10))/niter

	if save_name is not None or visualize:
		ax1 = plt.subplot(211)
		ax1.hist(p_vals, label="p10 = {}\nchain sample size = {}\ncompare samp size = {}".format(100 * p_10_ratio, len(chains)*chain_samples_scaling, compare_sample_size))
		ax1.set_title("p vals")
		ax1.legend()
		ax2 = plt.subplot(212)
		ax2.hist(d_vals)
		ax2.set_title("d vals")
		plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
	if save_name is not None:
		print "Saving figure to {}".format(save_name)
		plt.savefig(save_name)

	if visualize:	
		plt.show()
	plt.close('all')

	if verify_sampling:
		if save_name is not None or visualize:
			plt.hist(chains_0, histtype='step', label="Sampling", normed=True, range=(chains[0].min(), chains[0].max()))
			plt.hist(chains[0], histtype='step', label="Data", normed=True, range=(chains[0].min(), chains[0].max()))
			plt.title("First Chain Data and Sampling")
			plt.legend()
		if save_name is not None:
			sampling_save_name = save_name.split(".png")[0] + "_sampling_check.png"
			print "Saving figure to {}".format(sampling_save_name)
			plt.savefig(sampling_save_name)
		if visualize:
			plt.show()
	plt.close('all')
	return p_10_ratio


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--chains_file", help="The pickle file of cosi chains to sample over.", required=True)
	parser.add_argument("--niter", type=int, help="The number of iterations to run the simulation for.", default=1000)
	parser.add_argument("--dist_type", default='uniform', help="The distribution to compare against.")
	parser.add_argument("--mean", help="The mean of the normal distribution to compare.")
	parser.add_argument("--scale", help="The scale (or standard deviation) of the normal distribution to compare.")	
	parser.add_argument("--compare_sample_size", type=int, help="The number of samples to be drawn from the comparison distribution during each iteration of the simulation.")

	args = parser.parse_args()

	chains_file = args.chains_file
	niter = args.niter	
	dist_type = args.dist_type	
	mean_val = args.mean
	scale_val = args.scale
	compare_sample_size = args.compare_sample_size
	
	if dist_type == 'normal' and (mean_val is None or scale_val is None):
		raise Exception("Must use both --mean and --scale if --dist_type is 'normal'")

	try:
		chains, _ = pickle.load(open(chains_file, "rb"))
	except:
		chains = pickle.load(open(chains_file, "rb"))
	
	run_simulation_ad_ksamp(chains, (0, 1), distribution_type=dist_type, niter=niter, chain_samples_scaling=1, compare_sample_size=compare_sample_size, save_name=None, verify_sampling=False, visualize=True, mean=mean_val, scale=scale_val)

#	for n in [30, 100, 1000, 10000]:
#		p_vals = np.zeros(niter)
#		for i in range(niter):
#	                curr_sample = sample(chains)
#			_, _, p = ksamp([curr_sample, np.random.uniform(0, 1, n)])
#			p_vals[i] = p
#		p_10 = [p for p in p_vals if p < 0.1]
#		p_10_val = float(len(p_10))/niter
#		_ = plt.hist(p_vals, bins=50, range=(0, 1), histtype='step', label="Size = {} p10 = {}".format(n, 100 * p_10_val))
#		
#
#	plt.legend()
#	plt.savefig("/users/sstetzle/p_vals.png")
#	plt.show()

#	for sample_size in [30, 100, 1000, 10000, 10000]:
#		reject_ratio = run_simulation_ad_ksamp(chains, niter, compare_sample_size=sample_size)
#		print "Compare sample size {}".format(sample_size)
#		print "\tAD Test: {}% with p below 10%\n".format(100. * reject_ratio)
		

if __name__ == '__main__':
	main()

