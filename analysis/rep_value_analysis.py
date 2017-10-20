import numpy as np
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams.update({'font.size': 28})
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from anderson_darling import anderson_darling as ad
from scipy.stats import anderson_ksamp as ad_ksamp
import os

def get_step_ecdf(data):
	data.append(0.)
	data.append(1.)
#	sorted_data = np.sort(data)
#	y = np.arange(len(sorted_data)) / float(len(sorted_data))
	ecdf = ECDF(data)
	x = np.sort(data)
	y = ecdf(x)
	del data[-1]
	del data[-1]
	return x, y


def get_modes_gradient(list_of_chains):
	modes = []
	for chain in list_of_chains:
		ecdf = ECDF(chain)
		x = np.linspace(0, 1, 1000)
		y = ecdf(x)
		deriv = np.gradient(y)
		mode = x[np.argmax(deriv)]
		modes.append(mode)
	return modes


def get_modes(list_of_chains):
	modes = []
	for chain in list_of_chains:
		heights, bins = np.histogram(chain, bins=80, range=(0, 1))
		max_idx = np.argmax(heights)
		val_at_max = bins[max_idx] + (bins[1] - bins[0])/2.
		modes.append(val_at_max)
	return modes


def get_medians(list_of_chains):
	medians = [np.median(chain) for chain in list_of_chains]
	return medians


def get_means(list_of_chains):
	means = [np.mean(chain) for chain in list_of_chains]
	return means


def compare_ecdf(x_m2, y_m2, x_stig, y_stig, save_name):	
        uniform = lambda x : x
        x = np.linspace(0, 1, 10000)
        y = uniform(x)

	fig, axes = plt.subplots(1, 2, figsize=(14, 10))
	ax1 = axes[0]
	ax2 = axes[1]
	
	heights_m2, _, _ = ax1.hist(x_m2, bins=10, range=(0, 1), label="Data", normed=True)
	ax1.hist(x, bins=10, range=(0, 1), label="Model", histtype='step', normed=True)
	ax1.set_title("Trad.")
	ax1.set_xlabel("COSI value")
	ax1.legend()

	heights_stig, _, _ = ax2.hist(x_stig, bins=10, range=(0, 1), label="Data", normed=True)
	ax2.hist(x, bins=10, range=(0, 1), label="Model", histtype='step', normed=True)
	ax2.set_title("Ortho.")
	ax2.set_xlabel("COSI value")
	ax2.get_yaxis().set_visible(False)
	ax2.legend()

	min_y = 0
	max_y = max([max(heights_m2), max(heights_stig)])
	max_y += max_y * 0.10
	ax1.set_ylim([min_y, max_y])
	ax2.set_ylim([min_y, max_y])

	plt.tight_layout()

	save_file = "{}_{}.png".format(save_name, "hist_compare")
	print "Saving figure to {}".format(save_file)
	plt.savefig(save_file)

	plt.show()
	
	fig, axes = plt.subplots(1, 2, figsize=(14, 10))
	ax1 = axes[0]
	ax2 = axes[1]

        ax1.step(x_m2, y_m2, label="Data")
        ax1.plot(x, y, label="Model")
	ax1.set_title("Trad.")
	ax1.set_xlabel("COSI value")
	ax1.set_ylabel("Cummulative Probability")
	ax1.legend()

        ax2.step(x_stig, y_stig, label="Data")
        ax2.plot(x, y, label="Model")
	ax2.set_title("Ortho.")
	ax2.set_xlabel("COSI value")
	ax2.get_yaxis().set_visible(False)
	ax2.legend()

	plt.tight_layout()

	save_file = "{}_{}.png".format(save_name, "cdf_compare")
	print "Saving figure to {}".format(save_file)
	plt.savefig(save_file)

        #plt.show()


def main():
	save_dir = "/users/sstetzle/stat_analysis"

        pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))

        cosi_chains_m2 = []

        for pulsar in pulsar_dicts_m2.keys():
                _, _, par_dict = pulsar_dicts_m2[pulsar]
                cosi_chains_m2.append(par_dict['COSI'])


        pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

        cosi_chains_stig = []

        for pulsar in pulsar_dicts_stig.keys():
                _, _, par_dict = pulsar_dicts_stig[pulsar]
                cosi_chains_stig.append(par_dict['COSI'])

	cosi_means_m2 = get_means(cosi_chains_m2)
	cosi_means_stig = get_means(cosi_chains_stig)
	print "\nMeans STIG:"
	for pulsar, mean in zip(pulsar_dicts_stig.keys(), cosi_means_stig):
		print "{} {}".format(pulsar, mean)
	print "\nMeans M2:"
	for pulsar, mean in zip(pulsar_dicts_m2.keys(), cosi_means_m2):
		print "{} {}".format(pulsar, mean)

#	ax1 = plt.subplot(121)
#	ax1.hist(cosi_means_m2)
#	ax2 = plt.subplot(122)
#	ax2.hist(cosi_means_stig)
#	plt.show()

	cosi_medians_m2 = get_medians(cosi_chains_m2)
	cosi_medians_stig = get_medians(cosi_chains_stig)
	print "\nMedians STIG:"
	for pulsar, median in zip(pulsar_dicts_stig.keys(), cosi_medians_stig):
		print "{} {}".format(pulsar, median)
	print "\nMedians M2:"
	for pulsar, median in zip(pulsar_dicts_m2.keys(), cosi_medians_m2):
		print "{} {}".format(pulsar, median)
#
#	ax1 = plt.subplot(121)
#	ax1.hist(cosi_medians_m2)
#	ax2 = plt.subplot(122)
#	ax2.hist(cosi_medians_stig)
#	plt.show()

	cosi_modes_m2 = get_modes(cosi_chains_m2)
	cosi_modes_stig = get_modes(cosi_chains_stig)
	print "\nModes STIG:"
	for pulsar, mode in zip(pulsar_dicts_stig.keys(), cosi_modes_stig):
		print "{} {}".format(pulsar, mode)
	print "\nModes M2:"
	for pulsar, mode in zip(pulsar_dicts_m2.keys(), cosi_modes_m2):
		print "{} {}".format(pulsar, mode)

	cosi_modes_grad_m2 = get_modes_gradient(cosi_chains_m2)
	cosi_modes_grad_stig = get_modes_gradient(cosi_chains_stig)
	print "\nModes gradient STIG:"
	for pulsar, mode in zip(pulsar_dicts_stig.keys(), cosi_modes_grad_stig):
		print "{} {}".format(pulsar, mode)
	print "\nModes gradient M2:"
	for pulsar, mode in zip(pulsar_dicts_m2.keys(), cosi_modes_grad_m2):
		print "{} {}".format(pulsar, mode)


	x_m2, y_m2 = get_step_ecdf(cosi_means_m2)
	x_stig, y_stig = get_step_ecdf(cosi_means_stig)

	y_m2[np.where(x_m2 == 0)] = 0
	y_stig[np.where(x_stig == 0)] = 0

	_, p_m2 = ad(cosi_means_m2)
	_, p_stig = ad(cosi_means_stig)
	print "Means"
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	_, _, p_m2 = ad_ksamp([cosi_means_m2, np.random.uniform(0, 1, 1000000)])
	_, _, p_stig = ad_ksamp([cosi_means_stig, np.random.uniform(0, 1, 1000000)])
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	print ""

	compare_ecdf(x_m2, y_m2, x_stig, y_stig, os.path.join(save_dir, "means"))

	x_m2, y_m2 = get_step_ecdf(cosi_medians_m2)
	x_stig, y_stig = get_step_ecdf(cosi_medians_stig)

	_, p_m2 = ad(cosi_medians_m2)
	_, p_stig = ad(cosi_medians_stig)
	print "Medians"
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	_, _, p_m2 = ad_ksamp([cosi_medians_m2, np.random.uniform(0, 1, 1000000)])
	_, _, p_stig = ad_ksamp([cosi_medians_stig, np.random.uniform(0, 1, 1000000)])
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	print ""

	y_m2[np.where(x_m2 == 0)] = 0
	y_stig[np.where(x_stig == 0)] = 0

	compare_ecdf(x_m2, y_m2, x_stig, y_stig, os.path.join(save_dir, "medians"))

	x_m2, y_m2 = get_step_ecdf(cosi_modes_m2)
	x_stig, y_stig = get_step_ecdf(cosi_modes_stig)

	_, p_m2 = ad(cosi_modes_m2)
	_, p_stig = ad(cosi_modes_stig)
	print "Modes"
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	_, _, p_m2 = ad_ksamp([cosi_modes_m2, np.random.uniform(0, 1, 1000000)])
	_, _, p_stig = ad_ksamp([cosi_modes_stig, np.random.uniform(0, 1, 1000000)])
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	print ""

	y_m2[np.where(x_m2 == 0)] = 0
	y_stig[np.where(x_stig == 0)] = 0

	compare_ecdf(x_m2, y_m2, x_stig, y_stig, os.path.join(save_dir, "modes"))
	
	x_m2, y_m2 = get_step_ecdf(cosi_modes_grad_m2)
	x_stig, y_stig = get_step_ecdf(cosi_modes_grad_stig)

	_, p_m2 = ad(cosi_modes_grad_m2)
	_, p_stig = ad(cosi_modes_grad_stig)
	print "Modes Gradient"
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	_, _, p_m2 = ad_ksamp([cosi_modes_grad_m2, np.random.uniform(0, 1, 1000000)])
	_, _, p_stig = ad_ksamp([cosi_modes_grad_stig, np.random.uniform(0, 1, 1000000)])
	print "P-Values:\nM2: {} STIG: {}".format(p_m2, p_stig)
	print ""

	y_m2[np.where(x_m2 == 0)] = 0
	y_stig[np.where(x_stig == 0)] = 0

	compare_ecdf(x_m2, y_m2, x_stig, y_stig, os.path.join(save_dir, "modes_grad"))



#	ax1 = plt.subplot(121)
#	ax1.hist(cosi_modes_m2)
#	ax2 = plt.subplot(122)
#	ax2.hist(cosi_modes_stig)
#	plt.show()
	

if __name__ == "__main__":
	main()

