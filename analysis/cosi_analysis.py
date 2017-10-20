import argparse
import numpy as np
from matplotlib import pyplot as plt
from analysis_utils import get_pulsar_dict
import cPickle as pickle
import os
from ad_simulations import run_simulation_ad_ksamp

def sample(chains):
	ret = np.zeros(len(chains))
	for i, chain in enumerate(chains):
		idx = np.random.uniform(0, len(chain) - 1)
		ret[i] = chain[idx]
	return ret


def anderson_darling_analysis(pulsar_dicts, test_pulsars='all', save_dir=None, visualize=False):
	cosi_chains = []
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()
	
	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			cosi_chain = par_dict['COSI']
			cosi_chains.append(cosi_chain)
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)

	if save_dir is not None:	
		save_name = os.path.join(save_dir, "anderson_darling.png")
	else:
		save_name = None

	run_simulation_ad_ksamp(cosi_chains, par_range=(0, 1), distribution_type='uniform', niter=10000, save_name=save_name, verify_sampling=True, visualize=visualize)


# The number of samples for this method may be too low - histogramming will be very sensitive to the number of bins
def chi2_analysis_sample(pulsar_dicts, niter=10000, test_pulsars='all', nbins=5, save_dir=None, visualize=False):
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()
	
	plot = visualize or save_dir is not None

	cosi_chains = []
	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			cosi_chain = par_dict['COSI']
			cosi_chains.append(cosi_chain)
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)

	for i in range(niter):
		chains_sample = sample(cosi_chains)

		weights = np.ones_like(chains_sample) / float(len(chains_sample))

		if plot:
			bin_heights, bins, _ = plt.hist(cosi_chain, bins=nbins, range=(0, 1), weights=weights)
		else:
			bin_heights, bins = np.histogram(cosi_chain, bins=nbins, range=(0, 1), weights=weights)	
	
		observed_frequency = bin_heights
		


# This method actually makes no sense: a good measurement will appear narrow in the final histogram
def chi2_analysis(pulsar_dicts, test_pulsars='all', nbins=100, save_dir=None, visualize=False):
	if test_pulsars == 'all':
		test_pulsars = pulsar_dicts.keys()

	bin_heights = []
	
	plot = visualize or save_dir is not None

	cosi_chains = []

	for pulsar in test_pulsars:
		try:
			_, _, par_dict = pulsar_dicts[pulsar]
		except KeyError:
			print "{} not in pulsar dict - excluding.".format(pulsar)
			continue

		try:
			cosi_chain = par_dict['COSI']
		except KeyError:
			print "COSI not in {} par dictionary - exclusing.".format(pulsar)
		cosi_chains.append(cosi_chain)		

	for cosi_chain in cosi_chains:
		weights = np.ones_like(cosi_chain) / float(len(cosi_chain) * len(cosi_chains))
		if plot:
			heights, bins, _ = plt.hist(cosi_chain, bins=nbins, range=(0, 1), weights=weights)
		else:
			heights, bins = np.histogram(cosi_chain, bins=nbins, range=(0, 1), weights=weights)	
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

	args = parser.parse_args()

	pulsar_dir = args.pulsar_dir
	pulsar_dicts = args.pulsar_dicts
	visualize = args.visualize
	save_dir = args.save_dir
	test_pulsar_files = args.test_pulsar_files

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
		chi2_analysis(pulsar_dict, test_pulsars='all', save_dir=test_save_dir, visualize=visualize)
		anderson_darling_analysis(pulsar_dict, test_pulsars='all', save_dir=test_save_dir, visualize=visualize)
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
			chi2_analysis(pulsar_dict, test_pulsars=test_pulsars, save_dir=test_save_dir, visualize=visualize)
			anderson_darling_analysis(pulsar_dict, test_pulsars=test_pulsars, save_dir=test_save_dir, visualize=visualize)


if __name__ == "__main__":
	main()

