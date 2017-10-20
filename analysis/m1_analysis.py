import argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from analysis_utils import get_pulsar_dict, get_par_range
import cPickle as pickle
import os
from ad_simulations import run_simulation_ad_ksamp

def sample(chains):
	ret = np.zeros(len(chains))
	for i, chain in enumerate(chains):
		idx = np.random.uniform(0, len(chain) - 1)
		ret[i] = chain[idx]
	return ret


def anderson_darling_analysis(pulsar_dicts, niter=10000, nsample=1, compare_nsample=None, mean=1.46, scale=0.21, test_pulsars='all', save_dir=None, visualize=False):
	m1_chains = []
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()
	
	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			m1_chain = par_dict['M1']
			m1_chains.append(m1_chain)
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)
	if compare_nsample is None:
		compare_nsample = len(m1_chains) * nsample

        kwargs = {}
        kwargs['mean'] = mean
        kwargs['scale'] = scale

	par_range = get_par_range([pulsar_dicts], 'M1')

	if save_dir is not None:	
		save_name = os.path.join(save_dir, "anderson_darling_{}_chain_{}_compare.png".format(nsample, compare_nsample))
	else:
		save_name = None

	run_simulation_ad_ksamp(m1_chains, par_range=par_range, distribution_type='normal', niter=niter, chain_samples_scaling=nsample, compare_sample_size=compare_nsample, save_name=save_name, verify_sampling=False, visualize=visualize, **kwargs)


# The number of samples for this method may be too low - histogramming will be very sensitive to the number of bins
def chi2_analysis_sample(pulsar_dicts, niter=10000, test_pulsars='all', nbins=5, save_dir=None, visualize=False):
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()
	
	plot = visualize or save_dir is not None

	m1_chains = []
	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			m1_chain = par_dict['COSI']
			m1_chains.append(m1_chain)
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)

	for i in range(niter):
		chains_sample = sample(m1_chains)

		weights = np.ones_like(chains_sample) / float(len(chains_sample))

		if plot:
			bin_heights, bins, _ = plt.hist(m1_chain, bins=nbins, range=(0, 1), weights=weights)
		else:
			bin_heights, bins = np.histogram(m1_chain, bins=nbins, range=(0, 1), weights=weights)	
	
		observed_frequency = bin_heights
		


# This method actually makes no sense: a good measurement will appear narrow in the final histogram
def chi2_analysis(pulsar_dicts, test_pulsars='all', nbins=100, save_dir=None, visualize=False):
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()

	bin_heights = []
	
	plot = visualize or save_dir is not None

	m1_chains = []

	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			m1_chain = par_dict['COSI']
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)
		m1_chains.append(m1_chain)		

	for m1_chain in m1_chains:
		weights = np.ones_like(m1_chain) / float(len(m1_chain) * len(m1_chains))
		if plot:
			heights, bins, _ = plt.hist(m1_chain, bins=nbins, range=(0, 1), weights=weights)
		else:
			heights, bins = np.histogram(m1_chain, bins=nbins, range=(0, 1), weights=weights)	
		bin_heights.append(heights)

	observed_frequencies = np.zeros(nbins)
	for heights in bin_heights:
		observed_frequencies += heights
	
	expected_frequencies = np.ones_like(observed_frequencies) / float(nbins)
	
	relative_square_diff = [ (obs - exp)**2 / exp for obs, exp in zip(observed_frequencies, expected_frequencies)]
	
	chi_2 = sum(relative_square_diff)
	print "Computed chi squared = {} for {}\n\t{}".format(chi_2, save_dir.split("/")[-1], "\n\t".join(test_pulsars))
	
	if save_dir is not None:
		save_name = os.path.join(save_dir, "histograms.png")
		print "Saving figure to {}".format(save_name)
		plt.savefig(save_name)

	if visualize:
		plt.show()

	plt.close('all')

	if plot:
		x = np.linspace(0, 1, len(observed_frequencies))
		plt.plot(x, observed_frequencies, label="Observations")
		plt.plot(x, expected_frequencies, label="Expectation")
		plt.legend()
		if save_dir is not None:
			save_name = os.path.join(save_dir, "chi2_comparison.png")
			print "Saving figure to {}".format(save_name)
			plt.savefig(save_name)
		if visualize:
			plt.show()

	plt.close('all')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pulsar_dir", help="The directory containing pulsar directories with completed simulations to analyze")
	parser.add_argument("--pulsar_dicts", help="The pickle file containing pulsar dictionaries to be analyzed")
	parser.add_argument("--visualize", action="store_true", help="Include to show histograms.")
	parser.add_argument("--save_dir", help="The directory to save plots to.")
	parser.add_argument("--test_pulsar_files", nargs='+', help="A text file containing the name of one pulsar to test on each line.")
	parser.add_argument("--niter", type=int, default=10000, help="The number of iterations to run simulations for.")
	parser.add_argument("--nsample", type=int, default=1, help="The number of times to sample each chain before comparison.")
	parser.add_argument("--compare_nsample", type=int, default=None, help="The number of samples to draw from the comparison distribution before testing.")

	args = parser.parse_args()

	pulsar_dir = args.pulsar_dir
	pulsar_dicts = args.pulsar_dicts
	visualize = args.visualize
	save_dir = args.save_dir
	test_pulsar_files = args.test_pulsar_files
	niter = args.niter
	nsample = args.nsample
	compare_nsample = args.compare_nsample

	if (pulsar_dir is None and pulsar_dicts is None) or (pulsar_dir is not None and pulsar_dicts is not None):
		print "Must pass either a directory with pulsar simulations or a pickle file with pre-compiled pulsar dictionaries"
		exit()

	if pulsar_dir is not None:
		pulsar_dict = get_pulsar_dict(pulsar_dir)
	else:
		pulsar_dict, sim_type = pickle.load(open(pulsar_dicts, "rb"))

	if test_pulsar_files is None:
		if save_dir is not None:
			test_save_dir = os.path.join(save_dir, sim_type, 'all_pulsars')
			if not os.path.exists(test_save_dir):
				print "Making directory {}".format(test_save_dir)
				os.makedirs(test_save_dir)
		else:
			test_save_dir = None
		anderson_darling_analysis(pulsar_dict, test_pulsars='all', save_dir=test_save_dir, visualize=visualize, niter=niter, nsample=nsample, compare_nsample=compare_nsample)
	else:
		for test_pulsar_file in test_pulsar_files:
			test_pulsars = open(test_pulsar_file, 'r').readlines()
			test_pulsars = [p.strip() for p in test_pulsars if ('J' in p or 'B' in p) and ('+' in p or '-' in p)]
			test_pulsar_descriptor = test_pulsar_file.split('.')[0]
			if save_dir is not None:
				test_save_dir = os.path.join(save_dir, sim_type, test_pulsar_descriptor)
				if not os.path.exists(test_save_dir):
					print "Making directory {}".format(test_save_dir)
					os.makedirs(test_save_dir)
			else:
				test_save_dir = None
			anderson_darling_analysis(pulsar_dict, test_pulsars=test_pulsars, save_dir=test_save_dir, visualize=visualize, niter=niter, nsample=nsample, compare_nsample=compare_nsample)


if __name__ == "__main__":
	main()

