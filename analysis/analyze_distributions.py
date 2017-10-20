from analysis_utils import *
import argparse
from glob import glob
import os
import shutil
from scipy import stats
from stats import get_cdf
from ks_simulations import run_simulation_ks_two_samp
from ad_simulations import run_simulation_ad_ksamp
import cPickle as pickle
import acor.acor

def make_distribution_plots(pulsar_dict, pars, out_dir, burn_in=10000, visualize=False):
	for pulsar in pulsar_dict:
		print "\nWorking on pulsar {0}".format(pulsar)
		_, _, par_dict = pulsar_dict[pulsar]
	
		for par in par_dict:
			if par not in pars:
				continue

			print ""

			if out_dir is not None:
				out_filename = os.path.join(out_dir, "{0}_values.txt".format(par))
				print "Writing values to {0}".format(out_filename)
				out_file = open(out_filename, 'a')
				pulsar_dir = os.path.join(out_dir, pulsar)
				if not os.path.exists(pulsar_dir):
					print "Making directory {0}".format(pulsar_dir)
					os.makedirs(pulsar_dir)
	
				dist_plot_save = os.path.join(pulsar_dir, "{0}_{1}.png".format(pulsar, par))
				autocorr_plot_save = os.path.join(pulsar_dir, "{0}_{1}_autocorrelation.png".format(pulsar, par))
				trace_plot_save = os.path.join(pulsar_dir, "{}_{}_trace.png".format(pulsar, par))
			else:
				dist_plot_save = None
				autocorr_plot_save = None
				trace_plot_save = None

			chain = par_dict[par]
			num_iterations = len(chain)
			if len(chain) > burn_in:
				burn_chain = chain[burn_in:]
			else:
				burn_chain = chain
			burn_iterations = len(burn_chain)
			auto_correlation_time = acor.acor(burn_chain)[0]
			effective_samples = burn_iterations / auto_correlation_time

			lower, median, upper = get_median_and_bounds(par, chain, save_name=dist_plot_save, visual_check=visualize)
			make_autocorrelation_plot(par, chain, save_name=autocorr_plot_save, visualize=visualize)
			make_trace_plot(par, chain, save_name=trace_plot_save, visualize=visualize)

			print "{3}: {4} {0:.4f} (+ {1:.4f}  -{2:.4f})".format(median, upper - median, median - lower, pulsar, par)
			if out_dir is not None:
				out_file.write("{3}\t{0:.4f}\t{1:.4f}\t{2:.4f}\t{4}\t{5}\n".format(median, upper - median, median - lower, pulsar, num_iterations, effective_samples))


def make_pair_plots(pulsar_dict, pars, out_dir=None, visualize=False):
	for pulsar in pulsar_dict:
		_, _, par_dict = pulsar_dict[pulsar]
		for i, par_1 in enumerate(pars):
			for par_2 in pars[i+1:]:
				chain_1 = par_dict[par_1]
				chain_2 = par_dict[par_2]
				triplot_chains = np.swapaxes(np.asarray([chain_1, chain_2]), 0, 1)
				triplot_pars = [par_1, par_2]
				save_dir = os.path.join(out_dir, pulsar)
				if not os.path.exists(save_dir):
					print "Making directory {}".format(save_dir)
					os.makedirs(save_dir)
				make_triplot(triplot_chains, triplot_pars, visualize=visualize, save_dir=save_dir)
				

def stat_analysis_frequencies(pulsar_dicts, par_name, nbins=80, distribution_type='uniform', save_dir=None, visualize=False):
	accumulated_chain = None
	bin_heights_list = []
	par_chains = []
	
	if par_name == 'COSI' or par_name == 'SINI':
		par_range = (0, 1)
	elif par_name == 'KIN':
		par_range = (0, 90)
	else:
		par_range = get_par_range(pulsar_dicts, par_name)

	ax1 = plt.subplot(211)
	ax1.set_title("{} distributions".format(par_name))
	ax1.set_xlabel(par_name)

	for pulsar_dict in pulsar_dicts:
		for pulsar in pulsar_dict:
			_, _, par_dict = pulsar_dict[pulsar]
			if par_name in par_dict.keys():
				par_chain = par_dict[par_name]
				par_chains.append(par_chain)

				# Bin the relative frequencies of our data
				weights = np.ones_like(par_chain)/float(len(par_chain))
				bin_heights, bins, _ = ax1.hist(par_chain, bins=nbins, range=par_range, weights=weights)
				bin_heights_list.append(bin_heights)

	# Append each chain onto a large continuous chain
	# Change each chain to be the size of the smallest chain so that each pulsar simulation contributes equally to the data set
	# The smallest chain method should be changed in the future to an interpolation routine/downsampling instead of throwing away data
	chain_lengths = [len(par_chain) for par_chain in par_chains]
	smallest_chain_size = min(chain_lengths)
	for par_chain in par_chains:
		if accumulated_chain is not None:
			accumulated_chain = np.concatenate((accumulated_chain, par_chain[:smallest_chain_size]))
		else:
			accumulated_chain = par_chain[:smallest_chain_size]

	# Equally weight the relative frequencies of each of the pulsar chains
	accumulated_probability = np.zeros(nbins)
	for bin_heights in bin_heights_list:
		accumulated_probability += bin_heights
	accumulated_probability /= len(bin_heights_list)
	observed_frequencies = accumulated_probability# * len(accumulated_chain)

	if distribution_type == 'uniform':
		# Compare the observed frequencies for this simulation to an expectation of a uniform distribution
		expected_frequencies = np.ones_like(observed_frequencies) / float(nbins)  #* len(accumulated_chain) / float(nbins)

		# Compute the chi squared statistic
		relative_square_difference = [ (obs - exp)**2 / exp for obs, exp in zip(observed_frequencies, expected_frequencies) ]
		chi_squared = sum(relative_square_difference)# / len(accumulated_chain)

		# Compute the Kolmogorov-Smirnov statistic
		ks_stat, p_val = stats.kstest(observed_frequencies, 'uniform')
		sample_size = len(accumulated_chain)
#		x = np.linspace(par_range[0], par_range[1], sample_size)
		x = np.linspace(par_range[0], par_range[1], nbins)
		uniform_data = np.ones_like(x)

		ax2 = plt.subplot(212)
	#	ax2.plot(x, get_cdf(accumulated_chain))
		ax2.plot(x, get_cdf(observed_frequencies), label="ECDF")
		ax2.plot(x, get_cdf(uniform_data), label="Uniform")
		ax2.legend()
	else:
		print "{} is not implemented".format(distribution_type)
		chi_squared = -1
		ks_stat = -1		
		p_val = -1

	pulsars_string = " ".join([" ".join(pd.keys()) for pd in pulsar_dicts])
	par_distribution_string = "{} compared to {}".format(par_name, distribution_type)
	chi_squared_string = "chi squared = {}".format(chi_squared)
	ks_stat_string = "ks statistic = {}".format(ks_stat)
	p_val_string = "ks p value = {}".format(p_val)
	print_out = "\n\t".join([pulsars_string, par_distribution_string, chi_squared_string, ks_stat_string, p_val_string])
	print print_out

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	if save_dir is not None:
		save_name = "{}_pulsars_{}_compare_to_{}.png".format(len(pulsars_string.split(" ")), par_name, distribution_type)
		save_path = os.path.join(save_dir, save_name)
		print "Saving to {}".format(save_path)
		plt.savefig(save_path)

	if visualize:
		plt.show()

	plt.close('all')	


def stat_analysis_sampling(pulsar_dicts, par_name, niter=10000, chain_samples_scaling=1, num_compare_samples=30, distribution_type='uniform', save_dir=None, visualize=False):
	if par_name not in ['COSI', 'KIN'] and distribution_type in ['uniform']:
		print "Uniform statistical sampling with Anderson Darling test not implemented for {}".format(par_name)
		return

	if par_name == 'COSI' or par_name == 'SINI':
		par_range = (0, 1)
	elif par_name == 'KIN':
		par_range = (0, 90)
	else:
		par_range = get_par_range(pulsar_dicts, par_name)

	kwargs = {}
	if distribution_type == 'uniform':
		kwargs['min'] = par_range[0]
		kwargs['max'] = par_range[1]
	elif distribution_type == 'norm' and par_name == 'M1':
		kwargs['mean'] = 1.46
		kwargs['scale'] = 0.21
	else:
		print "Statistical sampling with Anderson Darling test not implemented for {} against distribution {}".format(par_name, distribution_type)
		return None

	par_chains = []
	for pulsar_dict in pulsar_dicts:
		for pulsar in pulsar_dict:
			_, _, par_dict = pulsar_dict[pulsar]
			if par_name in par_dict.keys():
				par_chain = par_dict[par_name]
				par_chains.append(par_chain)

	if save_dir is not None:
		save_name = os.path.join(save_dir, "{}_ad_sampling_{}.png".format(par_name, distribution_type))
	else:
		save_name = None	

	p10 = run_simulation_ad_ksamp(par_chains, par_range, distribution_type=distribution_type, niter=niter, chain_samples_scaling=chain_samples_scaling, save_name=save_name, verify_sampling=True, visualize=visualize, **kwargs)

	pulsars_string = " ".join([" ".join(pd.keys()) for pd in pulsar_dicts])
	par_distribution_string = "{} compared to {}".format(par_name, distribution_type)
	p_val_string = "{}% of p values below 10%".format(100*p10)
	print_out = "\n\t".join([pulsars_string, par_distribution_string, p_val_string])

	print print_out
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--visualize", action="store_true", help="Set to visualize plots as they are made")	
	parser.add_argument("--simulation_dirs", nargs='+', default=[], help="A list of directories containing all pulsars with chains to be analyzed")
	parser.add_argument("--simulation_files", nargs='+', action='store', default=[], help="The pickle files containing pulsar dictionaries for each simulation")
	parser.add_argument("--save_dir", action='store', required=True, help="The directory to save output to")
	parser.add_argument("--restrict_dists", action="store_true", help="Setting this restricts the analyzed distributions to COSI, M2, and M1")
	parser.add_argument("--clean", action="store_true", help="Set this option to clean the save directory before running")
	parser.add_argument("--pars_to_test", nargs='+', action="store", default=['COSI'], help="The list of parameters to test distributions for.")
	parser.add_argument("--distributions_to_test", nargs='+', action="store", default=['uniform'], help="The distribution you want to test each parameter distribution against.")
	parser.add_argument("--no_dist_plots", action="store_true", help="Include so that distribution plots are not produced.")
	parser.add_argument("--no_stat_analysis", action="store_true", help="Include so that no statistical analysis is performed.")
	parser.add_argument("--no_pair_plots", action="store_true", help="Include so that pair plots of each of the parameters of interest are not produced.")
	parser.add_argument("--niter", action="store", type=int, default=10000, help="The number of iterations to run the statistical p-value sampling for")

	args = parser.parse_args()
	sim_dirs = args.simulation_dirs
	save_dir = args.save_dir
	visualize = args.visualize	
	restrict_dists = args.restrict_dists
	clean = args.clean
	pars_to_test = args.pars_to_test
	distributions_to_test = args.distributions_to_test
	no_dist_plots = args.no_dist_plots
	no_stat_analysis = args.no_stat_analysis
	no_pair_plots = args.no_pair_plots
	sim_files = args.simulation_files
	niter = args.niter

	if len(sim_dirs) == 0 and len(sim_files) == 0:
		print "Either --simulation_dirs or --simulation_files must be specified to perform analysis."
		exit()

	if restrict_dists:
		pars_to_test = ['COSI', 'M2', 'M1']

	out_dir = save_dir

	if out_dir is not None:	
		if not os.path.exists(out_dir):
			print "Making directory {0}".format(out_dir)
			os.makedirs(out_dir)
		elif clean:
			rm_response = raw_input("Are you sure you would like to delete {0}? ".format(out_dir))
			while rm_response not in ['y', 'yes', 'Y', 'Yes', 'n', 'no', 'N', 'No']:
				rm_response = raw_input("Are you sure you would like to delete {0}? ".format(out_dir))
			if rm_response in ['y', 'yes', 'Y', 'Yes']:
				shutil.rmtree(out_dir)
				print "Making directory {0}".format(out_dir)
				os.makedirs(out_dir)	

	def make_dist_and_pair_plots(pulsar_dict, pars_to_test, save_dir, visualize=visualize):
		if not no_dist_plots:	
			make_distribution_plots(pulsar_dict, pars_to_test, save_dir, visualize=visualize)
		if not no_pair_plots:
			make_pair_plots(pulsar_dict, pars_to_test, save_dir, visualize=visualize)
		

	pulsar_dicts_sim_types = []
	for sim_dir in sim_dirs:
		pulsar_dict = get_pulsar_dict(sim_dir)
		sim_type = os.path.dirname(sim_dir).split('/')[-1]

		save_dir = os.path.join(out_dir, sim_type)
		make_directory(save_dir)

		pulsar_dicts_sim_types.append((pulsar_dict, sim_type))

		make_dist_and_pair_plots(pulsar_dict, pars_to_test, save_dir, visualize=visualize)

	for sim_file in sim_files:
		print "Opening {}".format(sim_file)
		pulsar_dict, sim_type = pickle.load(open(sim_file, "rb"))

		save_dir = os.path.join(out_dir, sim_type)
		make_directory(save_dir)

		pulsar_dicts_sim_types.append((pulsar_dict, sim_type))

		make_dist_and_pair_plots(pulsar_dict, pars_to_test, save_dir, visualize=visualize)
	
	if not no_stat_analysis:
		for par in pars_to_test:
			for distribution in distributions_to_test:
				for pulsar_dict, sim_type in pulsar_dicts_sim_types:
					stat_analysis_frequencies([pulsar_dict], par, save_dir=os.path.join(out_dir, sim_type), distribution_type=distribution, visualize=visualize)
					stat_analysis_sampling([pulsar_dict], par, save_dir=os.path.join(out_dir, sim_type), distribution_type=distribution, visualize=visualize, niter=niter)

				pulsar_dicts = [p[0] for p in pulsar_dicts_sim_types]
				stat_analysis_frequencies(pulsar_dicts, par, save_dir=out_dir, distribution_type=distribution, visualize=visualize)
				stat_analysis_sampling(pulsar_dicts, par, save_dir=out_dir, distribution_type=distribution, visualize=visualize, niter=niter)


if __name__ == "__main__":
	main()

