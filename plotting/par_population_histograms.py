import anderson_darling as ad
import cPickle as pickle
import matplotlib
matplotlib.rcParams.update({'font.size': 28})
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d
import argparse
import os


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("pars", nargs='+', help="The parameters to make plots for.")
	parser.add_argument("--pulsars_file", default=None, help="A file containing one pulsar per line. Plots will only be made for these pulsars.")

        args = parser.parse_args()

        pars = args.pars
	pulsars_file = args.pulsars_file

	if pulsars_file is not None:
		pulsars = open(pulsars_file, 'r').readlines()
		pulsars = [p.strip() for p in pulsars]
	else:
		pulsars = None

	pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))

	pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

	if pulsars is None:
		pulsars_m2 = pulsar_dicts_m2.keys()
		pulsars_stig = pulsar_dicts_stig.keys()
	else:
		pulsars_m2 = pulsars
		pulsars_stig = pulsars

	for par in pars:
		par_chains_m2 = []

		for pulsar in pulsars_m2:
			if pulsar == 'J0023+0923':
				continue
			_, _, par_dict = pulsar_dicts_m2[pulsar]
			par_chains_m2.append(par_dict[par])
		
		par_chains_stig = []

		for pulsar in pulsars_stig:
			if pulsar == 'J0023+0923':
				continue
			_, _, par_dict = pulsar_dicts_stig[pulsar]
			par_chains_stig.append(par_dict[par])

		if par in ['COSI', 'SINI']:
			par_range = (0, 1)
		elif par == 'KIN':
			par_range = (0, 90)
		else:
			min_stig = min([min(chain) for chain in par_chains_stig])
			max_stig = max([max(chain) for chain in par_chains_stig])
			min_m2 = min([min(chain) for chain in par_chains_m2])
			max_m2 = max([max(chain) for chain in par_chains_m2])

			min_x = min([min_stig, min_m2])
			max_x = max([max_stig, max_m2])
			par_range = (min_x, max_x)
			print "{} x range: {}".format(par, par_range)

		fig, axes = plt.subplots(1, 2, figsize=(14, 10))
		
		ax1 = axes[0]
		ax2 = axes[1]

		nbins = 100
		heights_m2 = []
		heights_stig = []
		for chain in par_chains_m2:
			weights = np.ones_like(chain) / float(len(chain) * len(par_chains_m2))
			height, binedges, _ = ax1.hist(chain, bins=nbins, range=par_range, weights=weights)
			heights_m2.append(height)

		for chain in par_chains_stig:
			weights = np.ones_like(chain) / float(len(chain) * len(par_chains_stig))
			height, binedges, _ = ax2.hist(chain, bins=nbins, range=par_range, weights=weights)
			heights_stig.append(height)

		ax1.set_xlabel("{} value".format(par))
		ax2.set_xlabel("{} value".format(par))

		ax1.set_title("Trad.")
		ax2.set_title("Ortho.")

		ax2.get_yaxis().set_visible(False)
		
		plt.tight_layout()
		par_dir = "/users/sstetzle/stat_analysis/{}".format(par)
		if not os.path.exists(par_dir):
			os.makedirs(par_dir)
		if pulsars is None:
			save_name = os.path.join(par_dir, "histograms_compare.png")
		else:
			save_name = os.path.join(par_dir, "histograms_compare_pulsar_subset.png")
		print "Saved figure to {}".format(save_name)
		plt.savefig(save_name)
		plt.show()

		# Create cummulative histogram
		accumulated_prob_m2 = np.zeros(nbins)
		accumulated_prob_stig = np.zeros(nbins)
		for i in range(len(par_chains_m2)):
			accumulated_prob_m2 += heights_m2[i]
		for i in range(len(par_chains_stig)):
			accumulated_prob_stig += heights_stig[i]

		# Smooth
		sigma = 1.15
		bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
		smoothed_m2 = filters.gaussian_filter(accumulated_prob_m2, sigma=sigma)
		smoothed_stig = filters.gaussian_filter(accumulated_prob_stig, sigma=sigma)
		y_vals_m2 = interp1d(bins, smoothed_m2, kind='cubic')
		y_vals_stig = interp1d(bins, smoothed_stig, kind='cubic')
		
		fig, axes = plt.subplots(1, 2, figsize=(14, 10))
		
		ax1 = axes[0]
		ax2 = axes[1]

		ax1.plot(bins, y_vals_m2(bins), 'b--', lw=4, zorder=2)
		ax2.plot(bins, y_vals_stig(bins), 'b--', lw=4, zorder=2)

		ax1.plot(bins, np.ones_like(bins) / float(len(bins)), 'r', lw=4, zorder=3)
		ax2.plot(bins, np.ones_like(bins) / float(len(bins)), 'r', lw=4, zorder=3)

		min_y = 0
		max_y = max([max(y_vals_stig(bins)), max(y_vals_m2(bins))])
		max_y += 0.10 * max_y

		ax1.set_ylim([min_y, max_y])
		ax2.set_ylim([min_y, max_y])

		ax1.set_xlabel("{} value".format(par))
		ax2.set_xlabel("{} value".format(par))

		ax1.set_title("Trad.")
		ax2.set_title("Ortho.")

		ax2.get_yaxis().set_visible(False)

		plt.tight_layout()
		par_dir = "/users/sstetzle/stat_analysis/{}".format(par)
		if not os.path.exists(par_dir):
			os.makedirs(par_dir)
		if pulsars is None:
			save_name = os.path.join(par_dir, "cummulative_histograms_compare.png")
		else:
			save_name = os.path.join(par_dir, "cummulative_histograms_compare_pulsar_subset.png")
		print "Saved figure to {}".format(save_name)
		plt.savefig(save_name)
		plt.show()


if __name__ == "__main__":
	main()
