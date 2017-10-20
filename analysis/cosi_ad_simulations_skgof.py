import anderson_darling as ad
from scipy.stats import uniform
from skgof import ad_test
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
matplotlib.rcParams.update({'font.size': 26})
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sample(chains):
	ret = np.zeros(len(chains))
	for i, chain in enumerate(chains):
		ret[i] = chain[int(np.random.uniform(0, len(chain) - 1))]
	return ret


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pulsars_file", default=None, help="A file containing one pulsar per line. Restrict simulations to just these pulsars.")
	args = parser.parse_args()
	
	pulsars_file = args.pulsars_file

	if pulsars_file is not None:
		pulsars = open(pulsars_file, 'r').readlines()
		pulsars = [p.strip() for p in pulsars]
		print "Pulsars:\n{}".format("\n".join(pulsars))
	else:
		pulsars = None


	pulsar_dicts_m2, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts_M2_SINI/pars_M2_SINI_pulsar_chain_dict.pkl", "rb"))

	cosi_chains_m2 = []

	if pulsars is not None:
		pulsar_list = pulsars
	else:
		pulsar_list = pulsar_dicts_m2.keys()

	for pulsar in pulsar_list:
		_, _, par_dict = pulsar_dicts_m2[pulsar]
		cosi_chains_m2.append(par_dict['COSI'])
	
	pulsar_dicts_stig, _ = pickle.load(open("/.lustre/aoc/students/sstetzle/long_simulations/dicts/pars_H3_STIG_pulsar_chain_dict.pkl", "rb"))

	cosi_chains_stig = []

	if pulsars is not None:
		pulsar_list = pulsars
	else:
		pulsar_list = pulsar_dicts_stig.keys()

	for pulsar in pulsar_list:
		_, _, par_dict = pulsar_dicts_stig[pulsar]
		cosi_chains_stig.append(par_dict['COSI'])

	niter = 10000
	p_vals_m2 = np.zeros(niter)
	p_vals_stig = np.zeros(niter)
	# Compare to uniform
	for i in range(niter):
#		A2, p = ad.anderson_darling(sample(cosi_chains_m2), cdf=lambda x : x)
		A2, p = ad_test(sample(cosi_chains_m2), uniform(0, 1))
		p_vals_m2[i] = p

#		A2, p = ad.anderson_darling(sample(cosi_chains_stig), cdf=lambda x : x)
		A2, p = ad_test(sample(cosi_chains_stig), uniform(0, 1))
		p_vals_stig[i] = p
	
	fig, axes = plt.subplots(1, 2, figsize=(14, 10))	
	ax1 = axes[0]
	ax2 = axes[1]
	
	p_10_m2 = len([p for p in p_vals_m2 if p < 0.1])/float(len(p_vals_m2))
	p_10_stig = len([p for p in p_vals_stig if p < 0.1])/float(len(p_vals_stig))

	heights_m2, bins, _ = ax1.hist(p_vals_m2, bins=20, label="{:0.1f}% under 0.10".format(100 * p_10_m2))
	heights_stig, bins, _ = ax2.hist(p_vals_stig, bins=20, label="{:0.1f}% under 0.10".format(100 * p_10_stig))

	ax1.legend()
	ax2.legend()

	min_y = 0
	max_y_m2 = max(heights_m2)
	max_y_m2 *= 1.25
	max_y_stig = max(heights_stig)
	max_y_stig *= 1.25

	ax1.set_title("Trad.")
	ax2.set_title("Ortho.")

	ax1.set_xlabel("p value")
	ax2.set_xlabel("p value")

	ax1.set_ylim([min_y, max_y_m2])
	ax2.set_ylim([min_y, max_y_stig])

	ax1.get_yaxis().set_visible(False)
	ax2.get_yaxis().set_visible(False)

	plt.tight_layout()
	if pulsars is None:
		save_name = "/users/sstetzle/stat_analysis/COSI/p_value_compare_skgof.png"
	else:
		save_name = "/users/sstetzle/stat_analysis/COSI/p_value_compare_pulsars_skgof.png"
	print "Saving figure to {}".format(save_name)
	plt.savefig(save_name)
	plt.show()



if __name__ == "__main__":
	main()
