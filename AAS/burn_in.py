import numpy as np
from psr_constants import Tsun
from psr_constants import SECPERDAY
from math import sqrt
import argparse
import os
from glob import glob
from analysis_utils import make_directory


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("sims_dir", help="The directory in which simulations were run.")
	parser.add_argument("out_dir", help="The directory to save the burned in and thinned chains.")
	parser.add_argument("--burn_in", default=0.25, type=float, help="The burn in parameter. Default = 0.25")
	parser.add_argument("--thinning", default=10, type=float, help="The thinning parameter. Default = 10")

	args = parser.parse_args()
	
	sims_dir = args.sims_dir
	out_dir = args.out_dir
	burn_in = args.burn_in
	thinning = args.thinning

	sim_name = sims_dir.split("/")[-2]
	print sim_name

	M2 = 'M2' in sim_name
	SINI = 'SINI' in sim_name
	
	H3 = 'H3' in sim_name
	H4 = 'H4' in sim_name

	STIG = 'STIG' in sim_name

	if M2 and SINI:
		idx_name_1 = 'M2'
		idx_name_2 = 'SINI'
	elif H3 and H4:
		idx_name_1 = 'H3'
		idx_name_2 = 'H4'
	elif H3 and STIG:
		idx_name_1 = 'H3'
		idx_name_2 = 'STIG'
	else:
		print "Neither M2, SINI, H3, STIG, nor H4 present in simulation directory name."

	
	pulsars = glob(os.path.join(sims_dir, "*"))
	for pulsar in pulsars:
		pulsar_name = os.path.basename(pulsar)

		chain_dir = os.path.join(pulsar, "chains")

		par_file = os.path.join(chain_dir, "pars.txt")
		chain_file = os.path.join(chain_dir, "chain_1.txt")

		if not os.path.exists(chain_file) or not os.path.exists(par_file):
			continue

		pars = np.genfromtxt(par_file, dtype=str)

		idx_1 = np.where(pars == idx_name_1)[0][0]
		idx_2 = np.where(pars == idx_name_2)[0][0]
		
		save_dir = os.path.join(out_dir, sim_name, pulsar_name)
		make_directory(save_dir)

		with open(os.path.join(out_dir, sim_name, pulsar_name, "chain_burn_{}_thin_{}.txt".format(burn_in, thinning)), "w") as outfile:
			print "Processing pulsar {}".format(pulsar_name)
			num_lines = 0
			with open(chain_file, 'r') as infile:
				for line in infile:
					num_lines += 1
			print num_lines

			start = int(num_lines * burn_in)
			with open(chain_file, "r") as infile:
				i = 0
				for line in infile:
					i += 1
					if i < start:
						continue
					if i % thinning != 0:
						continue

					data = np.fromstring(line, sep=' ', dtype=float)

					val_1 = data[idx_1]
					val_2 = data[idx_2]

					if M2 and SINI:
						sini = val_2	
						cosi = sqrt(1. - sini**2.)
					elif H3 and STIG:
						h3 = val_1
						stig = val_2
						sini = 2. * stig / (1. + stig**2.)
						cosi = sqrt(1. - sini**2.)
					elif H3 and H4:
						h3 = val_1
						h4 = val_2
						stig = h4 / h3
						sini = 2. * stig / (1. + stig**2.)
						cosi = sqrt(1. - sini**2.)

					outfile.write("{} {} {}\n".format(val_1, val_2, cosi))


if __name__ == "__main__":
	main()



#with open('m2_sini_out.txt', 'w') as outfile:
#	with open(chain_file, 'r') as infile:
#		i = 0
#		for line in infile:
#			if i != thinning:
#				i += 1
#				continue
#			else:
#				i = 0
#				#print line
#				data = np.fromstring(line, sep=' ', dtype=float)
#				h3 = data[h3_idx]
#				h4 = data[h4_idx]
#				
#				stig = h4 / h3
#				sini = 2. * stig / (1. + stig**2.)
#				cosi = sqrt(1. - sini**2.)
#				outfile.write("{} {} {}\n".format(h3, h4, cosi))
#
#
#pars_file = "binary/pars_M2_SINI/B1953+29/chains/pars.txt"
#chain_file = "binary/pars_M2_SINI/B1953+29/chains/chain_1.txt"
#
#pars = np.genfromtxt(pars_file, dtype=str)
#num_pars = len(pars)
#
#h3_idx = np.where(pars == 'M2')[0][0]
#h4_idx = np.where(pars == 'SINI')[0][0]
#
#MEM = 1e6
#N = int(MEM / (num_pars * 8))
#i = 0
#
##with open(chain_file, 'r') as infile:
##	gen = islice(infile, N)
##	arr = np.genfromtxt(gen, dtype=None)
##	i += arr.shape[0]
##	print arr.shape
##	print arr.nbytes
##	print arr[:, np.where(pars == 'H3')[0][0]].shape
##	print arr[:, np.where(pars == 'H3')[0][0]].nbytes
##
##exit()
#
#num_lines = 0
#with open(chain_file, 'r') as infile:
#	for line in infile:
#		num_lines += 1
##	while True:
##		print "{}                  \r".format(i),
##		gen = islice(infile, N)
##		arr = np.genfromtxt(gen, dtype=None)
##		i += arr.shape[0]
##		if arr.shape[0] < N:
##			break	
#
#burn_in = 0.25
#thinning = 10
#
#start = int(num_lines * burn_in)
#
#with open('m2_sini_out.txt', 'w') as outfile:
#	with open(chain_file, 'r') as infile:
#		i = 0
#		for line in infile:
#			if i != thinning:
#				i += 1
#				continue
#			else:
#				i = 0
#				#print line
#				data = np.fromstring(line, sep=' ', dtype=float)
#				h3 = data[h3_idx]
#				h4 = data[h4_idx]
#				
#				stig = h4 / h3
#				sini = 2. * stig / (1. + stig**2.)
#				cosi = sqrt(1. - sini**2.)
#				outfile.write("{} {} {}\n".format(h3, h4, cosi))
#			
#
#
#
#
