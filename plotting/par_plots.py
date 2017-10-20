import cPickle as pickle
import matplotlib
matplotlib.rcParams.update({'font.size': 32})
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d
import argparse
import os

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("pars", nargs='+', help="The parameters to make plots for.")
	
	args = parser.parse_args()
	
	pars = args.pars

	pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))
	pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

	pulsars_m2 = pulsar_dicts_m2.keys()
	pulsars_stig = pulsar_dicts_stig.keys()

	for pulsar_m2 in pulsars_m2:
		for pulsar_stig in pulsars_stig:
			if pulsar_m2 == pulsar_stig:
				for par in pars:
					_, _, par_dict_m2 = pulsar_dicts_m2[pulsar_m2]
					_, _, par_dict_stig = pulsar_dicts_stig[pulsar_stig]

					try:
						par_chain_m2 = par_dict_m2[par]
						par_chain_stig = par_dict_stig[par]	
					except KeyError:
						print "Parameter {} not found in dictionary for pulsar {}.".format(par, pulsar_m2)
						continue				

					fig, axes = plt.subplots(1, 2, figsize=(14, 10))
					ax1 = axes[0]
					ax2 = axes[1]

					if par in ['COSI', 'SINI']:
						par_range = (0, 1)
					elif par == 'KIN':
						par_range = (0, 90)
					else:
						combined_chain = np.hstack((par_chain_m2, par_chain_stig))
						min_x = np.hstack(combined_chain).min()
						max_x = np.hstack(combined_chain).max()
						par_range = (min_x, max_x)

					nbins = 100
					weights = np.ones_like(par_chain_m2) / float(len(par_chain_m2))
					heights_m2, _, _ = ax1.hist(par_chain_m2, bins=nbins, range=par_range, weights=weights)

					weights = np.ones_like(par_chain_stig) / float(len(par_chain_stig))
					heights_stig, _, _ = ax2.hist(par_chain_stig, bins=nbins, range=par_range, weights=weights)
					
					min_y = 0
					max_y = max(list(heights_m2) + list(heights_stig))
					max_y += 0.10 * max_y

					ax1.set_xlabel("{} value".format(par))
					ax2.set_xlabel("{} value".format(par))

					ax1.set_title("Trad.")
					ax2.set_title("Ortho.")

					ax1.set_ylim([min_y, max_y])
					ax2.set_ylim([min_y, max_y])
	
					#ax1.get_yaxis().set_visible(False)
					ax2.get_yaxis().set_visible(False)
	
					plt.tight_layout()
					par_dir = "/users/sstetzle/stat_analysis/{}".format(par)
					if not os.path.exists(par_dir):
						os.makedirs(par_dir)
					save_name = os.path.join(par_dir, "{}_compare.png".format(pulsar_m2))
					print "Saved figure to {}".format(save_name)
					plt.savefig(save_name)
					# plt.show()
					plt.close('all')

					fig, axes = plt.subplots(1, 2, figsize=(14, 10))
					ax1 = axes[0]
					ax2 = axes[1]

					sigma = 1.25

					combined_chain = np.hstack((par_chain_m2, par_chain_stig))
					min_x = combined_chain.min()
					max_x = combined_chain.max()

					par_range = (min_x, max_x)

					weights = np.ones_like(par_chain_m2) / float(len(par_chain_m2))

					heights, binedges = np.histogram(par_chain_m2, bins=nbins, range=par_range, density=True)
					bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
					heights = filters.gaussian_filter(heights, sigma=sigma)
					f = interp1d(bins, heights, kind='cubic')

					ax1.plot(bins, f(bins), 'b--', lw=3, zorder=2)

					max_m2 = max(heights)

					weights = np.ones_like(par_chain_stig) / float(len(par_chain_stig))

					heights, binedges = np.histogram(par_chain_stig, bins=nbins, range=par_range, density=True)
					bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
					heights = filters.gaussian_filter(heights, sigma=sigma)
					f = interp1d(bins, heights, kind='cubic')

					max_stig = max(heights)

					ax2.plot(bins, f(bins), 'b--', lw=3, zorder=2)

					min_y = 0
					max_y = max([max_m2, max_stig])
					max_y += 0.10 * max_y

					ax1.set_xlabel("{} value".format(par))
					ax2.set_xlabel("{} value".format(par))

					ax1.set_title("Trad.")
					ax2.set_title("Ortho.")

					ax1.set_ylim([min_y, max_y])
					ax2.set_ylim([min_y, max_y])
	
					ax1.get_yaxis().set_visible(False)
					ax2.get_yaxis().set_visible(False)

					plt.tight_layout()
					par_dir = "/users/sstetzle/stat_analysis/{}".format(par)
					if not os.path.exists(par_dir):
						os.makedirs(par_dir)
					save_name = os.path.join(par_dir, "{}_compare_smooth.png".format(pulsar_m2))
					print "Saved figure to {}".format(save_name)
					plt.savefig(save_name)
					# plt.show()
					plt.close('all')



if __name__ == "__main__":
	main()
