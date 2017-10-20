from glob import glob
import os
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

def make_plot(ephem_1_pars, ephem_1_chain, ephem_2_pars, ephem_2_chain, par_name, pulsar, ephem_1, ephem_2, out_dir):
	ephem_1_sini_idx = np.where(ephem_1_pars == par_name)[0][0]
	ephem_1_chain = ephem_1_chain[:, ephem_1_sini_idx]

	ephem_2_sini_idx = np.where(ephem_2_pars == par_name)[0][0]
	ephem_2_chain = ephem_2_chain[:, ephem_2_sini_idx]

	min_x = min([min(x) for x in [ephem_1_chain, ephem_2_chain]])
	max_x = max([max(x) for x in [ephem_1_chain, ephem_2_chain]])
	

	plt.figure()
	ax1 = plt.subplot(211)
	ax1.hist(ephem_1_chain, 80)
	ax1.set_title("{} {} Dist".format(ephem_1, par_name))
	ax1.set_xlim(min_x, max_x)	
	
	ax2 = plt.subplot(212)
	ax2.hist(ephem_2_chain, 80)
	ax2.set_title("{} {} Dist".format(ephem_2, par_name))
	ax2.set_xlabel(par_name)
	ax2.set_xlim(min_x, max_x)	

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	save_dir = os.path.join(out_dir, pulsar)
	if not os.path.exists(save_dir):
		print "Making directory {}".format(save_dir)
		os.makedirs(save_dir)
	save_name = os.path.join(save_dir, "{}_vs_{}_{}.png".format(ephem_1, ephem_2, par_name))
	print "Saving plot to {}".format(save_name)
	plt.savefig(save_name)
	plt.close("all")



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("workingdir", help="The directory containing the par files of interest for the different ephemerides.")
	parser.add_argument("ephem_1", help="The first ephemeris to compare.")
	parser.add_argument("ephem_2", help="The second ephemeris to compare.")
	parser.add_argument("savedir", help="The directory to save plots to.")
	
	args = parser.parse_args()

	in_dir = args.workingdir
	ephem_1 = args.ephem_1
	ephem_2 = args.ephem_2
	out_dir = args.savedir

	ephem_1_dir = os.path.join(in_dir, "pars_{}".format(ephem_1))
	ephem_2_dir = os.path.join(in_dir, "pars_{}".format(ephem_2))

	if not os.path.exists(ephem_1_dir):
		print "{} does not exist!".format(ephem_1_dir)
		exit()
	if not os.path.exists(ephem_2_dir):
		print "{} does not exist!".format(ephem_2_dir)
		exit()
	
	par_types_1 = glob(os.path.join(ephem_1_dir, "*"))
	par_types_1 = [d for d in par_types_1 if os.path.basename(d).startswith("pars")]
#	print "Found par types in {}".format(ephem_1_dir)
#	print "\n".join(par_types_1)

	par_types_2 = glob(os.path.join(ephem_2_dir, "*"))
	par_types_2 = [d for d in par_types_2 if os.path.basename(d).startswith("pars")]
#	print "Found par types in {}".format(ephem_2_dir)
#	print "\n".join(par_types_2)
	
	par_types_1_names = set([os.path.basename(d) for d in par_types_1])
	par_types_2_names = set([os.path.basename(d) for d in par_types_2])

	common_par_types = par_types_1_names & par_types_2_names
	
#	print "Common par types:"
#	print "\n".join(common_par_types)

	for par_type in common_par_types:
		print "Comparing simulations for type {}".format(par_type)

		ephem_1_par_dir = [d for d in par_types_1 if os.path.basename(d) == par_type]
		ephem_1_par_dir = ephem_1_par_dir[0]
		
		ephem_1_pulsars = glob(os.path.join(ephem_1_par_dir, "*"))
		ephem_1_pulsars_ran = [d for d in ephem_1_pulsars if os.path.exists(os.path.join(d, "chains"))]
		print "Pulsars that ran for ephem 1:"
		print "\n".join(ephem_1_pulsars_ran)

		ephem_2_par_dir = [d for d in par_types_2 if os.path.basename(d) == par_type]
		ephem_2_par_dir = ephem_2_par_dir[0]
		
		ephem_2_pulsars = glob(os.path.join(ephem_2_par_dir, "*"))
		ephem_2_pulsars_ran = [d for d in ephem_2_pulsars if os.path.exists(os.path.join(d, "chains"))]
		print "Pulsars that ran for ephem 2:"
		print "\n".join(ephem_2_pulsars_ran)

		ephem_1_pulsars_ran_names = [os.path.basename(d) for d in ephem_1_pulsars_ran]
		ephem_2_pulsars_ran_names = [os.path.basename(d) for d in ephem_2_pulsars_ran]
	
		common_ran_pulsars = set(ephem_1_pulsars_ran_names) & set(ephem_2_pulsars_ran_names)
		print "Common ran pulsars:"
		print "\n".join(common_ran_pulsars)
		if len(common_ran_pulsars) == 0:
			print "No pulsars in were run in common between {} and {}".format(ephem_1, ephem_2)

		for pulsar in common_ran_pulsars:
			ephem_1_pulsar = [p for p in ephem_1_pulsars if os.path.basename(p) == pulsar]
			ephem_1_pulsar = ephem_1_pulsar[0]
			ephem_2_pulsar = [p for p in ephem_2_pulsars if os.path.basename(p) == pulsar]
			ephem_2_pulsar = ephem_2_pulsar[0]
			
			ephem_1_par_file = os.path.join(ephem_1_pulsar, "chains", "pars.txt")
			ephem_2_par_file = os.path.join(ephem_2_pulsar, "chains", "pars.txt")
			
			print "Opening:\n{}\n{}".format(ephem_1_par_file, ephem_2_par_file)
			ephem_1_pars = np.genfromtxt(ephem_1_par_file, dtype='str')
			ephem_2_pars = np.genfromtxt(ephem_2_par_file, dtype='str')

			ephem_1_chain = np.loadtxt(os.path.join(ephem_1_pulsar, "chains", "chain_1.txt"))
			ephem_2_chain = np.loadtxt(os.path.join(ephem_2_pulsar, "chains", "chain_1.txt"))

			if 'H3' in ephem_1_pars and 'H3' in ephem_2_pars:
				make_plot(ephem_1_pars, ephem_1_chain, ephem_2_pars, ephem_2_chain, 'H3', pulsar, ephem_1, ephem_2, out_dir)
			if 'M2' in ephem_1_pars and 'M2' in ephem_2_pars:
				make_plot(ephem_1_pars, ephem_1_chain, ephem_2_pars, ephem_2_chain, 'M2', pulsar, ephem_1, ephem_2, out_dir)
			if 'SINI' in ephem_1_pars and 'SINI' in ephem_2_pars:
				make_plot(ephem_1_pars, ephem_1_chain, ephem_2_pars, ephem_2_chain, 'SINI', pulsar, ephem_1, ephem_2, out_dir)
	


if __name__ == "__main__":
	main()
