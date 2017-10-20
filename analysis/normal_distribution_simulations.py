from scipy import exp
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import pyplot as plt
import cPickle as pickle
import sys
from analysis_utils import save_object


def sample_chains(chains):
	ret_samples = []
	for chain in chains:
		idx = int(np.random.uniform(0, len(chain) - 1))
		ret_samples.append(chain[idx])
	return np.asarray(ret_samples)


def gauss(x, a, x0, sigma):
	return a*exp(-(x-x0)**2/(2*sigma**2))


def run_simulation(chains, desc, niter=1000, nbins=20, visualize=False):
	gaussian_centers = np.zeros(niter)
	gaussian_dispersions = np.zeros(niter)

	for i in range(niter):
		if i % int(0.01 * niter) == 0:
			sys.stdout.write("\r{}% done   ".format(100.*float(i)/niter))
			sys.stdout.flush()
		sample = sample_chains(chains)
		y, bins = np.histogram(sample, bins=nbins)
		x = bins[:-1] + (bins[1] - bins[0]) / 2
		try:
			popt, pcov = curve_fit(gauss, x, y, p0=[1, np.mean(x), np.std(x)])
		except:
			if i != 0:
				i -= 1
			continue
		a, x0, sigma = popt
		gaussian_centers[i] = x0
		gaussian_dispersions[i] = sigma
		if visualize and i % (niter / 10) == 0:
			plt.close("all")
			plt.plot(x, y, 'b+', label='data')
			x_fit = np.arange(min(x), max(x), 0.001)
			plt.plot(x_fit, gauss(x_fit, a, x0, sigma), label='fit')
			plt.legend()
			plt.show()
			plt.close("all")

	sys.stdout.write("\r100% done   ")
	sys.stdout.flush()
	print ""

	if visualize:
		centers_range = (0.5, 2.5)
		ax1 = plt.subplot(211)
		ax1.set_title("Centers")
		ax1.hist(gaussian_centers, bins=120, range=centers_range)
		ax1.legend()
		dispersion_range = (0.001, 0.25)
		ax2 = plt.subplot(212)
		ax2.set_title("Disersions")
		ax2.hist(gaussian_dispersions, bins=120, range=dispersion_range)
		ax2.legend()
		plt.tight_layout()
		plt.show()
	
	save_object(gaussian_centers, "/.lustre/aoc/students/sstetzle/long_simulations/normal_distributions/centers_iter_{}_bins_{}_{}.pkl".format(niter, nbins, desc))
	save_object(gaussian_dispersions, "/.lustre/aoc/students/sstetzle/long_simulations/normal_distributions/dispersions_iter_{}_bins_{}_{}.pkl".format(niter, nbins, desc))
	
	
def main():
        test_pulsars = open("../stat_analysis/m1/good_pulsars.txt", "r").readlines()
        test_pulsars = [p.strip() for p in test_pulsars]
        print "Using pulsars {} for M1 analysis.".format(", ".join(test_pulsars))

        pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))

        m1_chains_m2 = []

        for pulsar in test_pulsars:
                _, _, par_dict = pulsar_dicts_m2[pulsar]
                m1_chains_m2.append(par_dict['M1'])


        pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

        m1_chains_stig = []

        for pulsar in test_pulsars:
                _, _, par_dict = pulsar_dicts_stig[pulsar]
                m1_chains_stig.append(par_dict['M1'])


	niter = 100000
	for bins in np.arange(10, 15):
		run_simulation(m1_chains_m2, 'm2_sini', niter=niter, nbins=bins)
		run_simulation(m1_chains_stig, 'h3_stig', niter=niter, nbins=bins)


if __name__ == '__main__':
	main()

