import numpy as np
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt
import cPickle as pickle
import argparse


def par_plot(pulsar_dict, pulsar_name, par_name, descr, nbins=80, sigma=1.25):
	_, _, par_dict = pulsar_dict[pulsar_name]
	par_chain = par_dict[par_name]

        vals, binedges = np.histogram(par_chain, bins=nbins, density=True)
        bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
        vals = filters.gaussian_filter(vals, sigma=sigma)
        f = interp1d(bins, vals, kind='cubic')

        pdf = f(bins)
        pdf = np.divide(pdf, np.sum(pdf))

        cdf = [np.sum(pdf[:i]) for i in range(len(pdf))]

        fig = plt.figure("pdf", figsize=(12, 8), dpi=100)

#        plt.hist(par_chain, nbins, normed=True, label="Data")
        plt.plot(bins, f(bins), 'b--', lw=3, zorder=2, label="Smoothed PDF")
        #plt.title("{0} PDF".format(par_name))
        plt.xlabel("{0} Value".format(par_name))
	fig.axes[0].get_yaxis().set_visible(False)
	if descr == 'M2':
		plt.title("Trad.")
	elif descr == 'STIG':
		plt.title("Ortho.")
        #plt.legend()
	save_name = "/users/sstetzle/stat_analysis/PDFs/{}_{}_{}_PDF.png".format(pulsar_name, par_name, descr)
	print "Saving figure to {}".format(save_name)
	plt.savefig(save_name)
	plt.show()


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("par", help="The parameter to make a distribution plot for.")
	parser.add_argument("pulsar", help="The pulsar to make a distribution plot for.")

	args = parser.parse_args()

	par = args.par
	pulsar = args.pulsar

        pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))
        pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

	par_plot(pulsar_dicts_m2, pulsar, par, 'M2')
	par_plot(pulsar_dicts_stig, pulsar, par, 'STIG')


if __name__ == "__main__":
	main()
