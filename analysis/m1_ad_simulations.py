import anderson_darling as ad
import cPickle as pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import norm

def sample(chains):
	ret = np.zeros(len(chains))
	for i, chain in enumerate(chains):
		ret[i] = chain[int(np.random.uniform(0, len(chain) - 1))]
	return ret


def main():
	test_pulsars = open("/users/sstetzle/stat_analysis/m1/good_pulsars.txt", "r").readlines()
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

	niter = 10000
	p_vals_m2 = np.zeros(niter)
	p_vals_stig = np.zeros(niter)
	# Compare to uniform
	for i in range(niter):
		chain_sample = sample(m1_chains_m2)
		A2, _, _, p = ad.anderson_darling_normal(chain_sample, mu=1.46, sigma=0.21)
		if math.isnan(p):
			i -= 1
			continue
		else:
			p_vals_m2[i] = p
			if i % (niter / 5) == 0:
				x = np.arange(0.26, 2.66, 0.001)
				y = norm.pdf(x, 1.46, 0.21)
				plt.plot(x, y)
				_ = plt.hist(chain_sample, bins=10, range=(0.26, 2.66), normed=True)
				save_name = "/users/sstetzle/stat_analysis/M1/ozel_compare_m2_sini_{}.png".format(int(i / (niter / 5)))
				print "Saving figure to {}".format(save_name)
				plt.savefig(save_name)
				plt.close('all')

	for i in range(niter):
		chain_sample = sample(m1_chains_stig)
		A2, _, _, p = ad.anderson_darling_normal(chain_sample, mu=1.46, sigma=0.21)
		if math.isnan(p):
			i -= 1
			continue
		else:
			p_vals_stig[i] = p
			if i % (niter / 5) == 0:
				x = np.arange(0.26, 2.66, 0.001)
				y = norm.pdf(x, 1.46, 0.21)
				plt.plot(x, y)
				_ = plt.hist(chain_sample, bins=10, range=(0.26, 2.66), normed=True)
                                save_name = "/users/sstetzle/stat_analysis/M1/ozel_compare_h3_stig_{}.png".format(int(i / (niter / 5)))
                                print "Saving figure to {}".format(save_name)
				plt.savefig(save_name)
				plt.close('all')
	
        fig, axes = plt.subplots(1, 2, figsize=(12, 8)) 
        ax1 = axes[0]
        ax2 = axes[1]
	
	p_10_m2 = len([p for p in p_vals_m2 if p < 0.1])/float(len(p_vals_m2))
	p_10_stig = len([p for p in p_vals_stig if p < 0.1])/float(len(p_vals_stig))

	_ = ax1.hist(p_vals_m2, bins=20, range=(0, 1), label="{}% under 0.10".format(100 * p_10_m2))
	_ = ax2.hist(p_vals_stig, bins=20, range=(0, 1), label="{}% under 0.10".format(100 * p_10_stig))

	ax1.legend()
	ax2.legend()

	ax1.set_title("M2 SINI")
	ax2.set_title("H3 STIG")

	ax1.set_xlabel("p value")
	ax2.set_xlabel("p value")

	plt.tight_layout()
	save_name = "/users/sstetzle/stat_analysis/M1/p_value_compare_ozel.png"
	print "Saving figure to {}".format(save_name)
	plt.savefig(save_name)
	plt.show()



if __name__ == "__main__":
	main()
