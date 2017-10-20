from glob import glob
import os
from binary_psr import *
from psr_utils import companion_mass
from psr_constants import Tsun
import sys
import shutil
import argparse
from math import sin, cos, sqrt
import numpy as np

def line_is_present(par_lines, par_name):
	par_line = [l for l in par_lines if l.startswith(par_name + " ")]
	present = len(par_line) == 1
	if present:
		try:
			par_val = float(par_line[0].split()[1].replace('D', 'e'))
		except:
			par_val = par_line[0].split()[1]
	else:
		par_val = None
	return present, par_val


def get_stig_from_h3_assume_M2(H3, M2=1.5):
	STIG = (H3 / shapR(M2))**(1./3.)
	return STIG

def get_stig_from_h3_assume_I_M1(binary, H3, I=60.0, M1=1.4):
	PB = binary.par.PB
	x = binary.par.A1
	M2 = companion_mass(PB, x, inc=I, mpsr=M1)
	STIG = (H3 / shapR(M2))**(1./3.)
	return STIG
	
def get_m2_and_sini_from_h3(H3, M2=0.5):
	r = shapR(M2)
	STIG = pow(H3/r, 1./3.)
	SINI = 2. * STIG / (1. + pow(STIG, 2.))
	M2 = H3 / pow(STIG, 3.) / 4.925490947e-6
	return M2, SINI

def get_m2_and_sini_from_h3_assume_I_M1(binary, H3, I=60.0, M1=1.4):
	PB = binary.par.PB
	x = binary.par.A1
	M2 = companion_mass(PB, x, inc=I, mpsr=M1)	
#	r = shapR(M2)
#	STIG = pow(H3/r, 1./3.)
#	SINI = 2. * STIG / (1. + pow(STIG, 2.))
	SINI = sin(I * np.pi/180.)
	return M2, SINI
	

def get_m2_and_sini_from_none_assume_I_M1(binary, I=60.0, M1=1.4):
	PB = binary.par.PB
	x = binary.par.A1
	M2 = companion_mass(PB, x, inc=I, mpsr=M1)
	SINI = sin(I * np.pi/180.)
	return M2, SINI


def get_h3_and_stig_from_none_assume_I_M1(binary, I=60.0, M1=1.4):
	PB = binary.par.PB
	x = binary.par.A1
	M2 = companion_mass(PB, x, inc=I, mpsr=M1)
	COSI = cos(I * np.pi/180.)
	STIG = sqrt(1. - COSI)/sqrt(1. + COSI)
	H3 = shapR(M2) * STIG**3.
	return H3, STIG


def get_h3_and_stig_from_m2_and_sini(binary):
	M2 = binary.par.M2
	SINI = binary.par.SINI
	COSI = sqrt(1. - SINI**2.)
	STIG = sqrt(1. - COSI)/sqrt(1. + COSI)
	H3 = shapR(M2) * STIG**3.
	return H3, STIG


def get_h3_and_stig_from_m2_and_kin(M2, KIN):
	SINI = np.sin(KIN * np.pi / 180.)
	COSI = sqrt(1. - SINI**2.)
	STIG = sqrt(1. - COSI)/sqrt(1. + COSI)
	H3 = shapR(M2) * STIG**3.
	return H3, STIG
	

def get_m2_and_sini_from_none_assume_M1_M2(binary, M1=1.35, M2=1.5):
	try:
		PB = binary.par.PB
	except:
		try:
			PB = 1./binary.par.FB
		except:
			PB = None
	if PB is None:
		return None, None
	SINI = shapS(M1, M2, binary.par.A1, binary.par.PB)
	return M2, SINI


def change_binary_model(par_lines, new_model):
	_, model = line_is_present(par_lines, "BINARY") 
	par_lines = [l for l in par_lines if not l.startswith("BINARY ")]
	if model == "ELL1H":
		par_lines = [l for l in par_lines if not l.startswith("H3 ")]
	par_lines.append("\nBINARY {}".format(new_model))
	return par_lines


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("pulsardir", help="The directory containing pulsar directories")
	parser.add_argument("savedir", help="The directory to save the new par files to")
	parser.add_argument("--h3_to_m2_sini", action="store_true", help="Use this option to convert H3 pulsars to M2 and SINI", required=False)
	args = parser.parse_args()

	in_dir = args.pulsardir
	out_dir = args.savedir
	h3_to_m2_sini = args.h3_to_m2_sini

	pulsars = glob(os.path.join(in_dir, "*"))
	for pulsar in pulsars:
		par_file = glob(os.path.join(pulsar, "*.par"))
		tim_file = glob(os.path.join(pulsar, "*.tim"))

		if len(par_file) == 0:
			print "Pulsar {} has no par files".format(os.path.basename(pulsar))
			continue

		if len(tim_file) == 0:
			print "Pulsar {} has no tim files".format(os.path.basename(pulsar))
			continue

		tim_file = tim_file[0]
		par_file = par_file[0]
		print "Opening {}".format(par_file)

		binary = binary_psr(par_file)
		binary_type = binary.par.BINARY

		par_lines = open(par_file, 'r').readlines()
		h3_present, H3 = line_is_present(par_lines, "H3")
		m2_present, M2 = line_is_present(par_lines, "M2")
		
		if h3_present:
			if h3_to_m2_sini:
				par_lines = change_binary_model(par_lines, "ELL1")
				#M2, SINI = get_m2_and_sini_from_h3(H3)
				M2, SINI = get_m2_and_sini_from_h3_assume_I_M1(binary, H3)

				print "Adding \"{}\" to par lines".format("SINI {} 1".format(SINI))
				par_lines.append("\nSINI {} 1\n".format(SINI))
		
				print "Adding \"{}\" to par lines".format("M2 {} 1".format(M2))
				par_lines.append("\nM2 {} 1\n".format(M2))
			else:
				STIG = get_stig_from_h3_assume_M2(H3, M2=0.5)
				print "Adding \"{}\" to par lines".format("STIG {} 1".format(STIG))
				par_lines.append("\nSTIG {} 1\n".format(STIG))

			out_dir_file = os.path.join(out_dir, os.path.basename(pulsar))
			if not os.path.exists(out_dir_file):
				print "Making directory {}".format(out_dir_file)
				os.makedirs(out_dir_file)
	
			to = os.path.join(out_dir_file, os.path.basename(tim_file))
			print "Copying {} --> {}".format(tim_file, to)
			shutil.copy2(tim_file, to)

			out_dir_file = os.path.join(out_dir_file, os.path.basename(par_file))
			print "Writing new par lines to {}".format(out_dir_file)
			out = open(out_dir_file, 'w')
			out.writelines(par_lines)

		elif 'DD' == binary_type or 'ELL1' == binary_type and not m2_present:
			# Chaning assumptions on masses
#			M2, SINI = get_m2_and_sini_from_none_assume_M1_M2(binary, M1=1.8, M2=1.2)
			M2, SINI = get_m2_and_sini_from_none_assume_I_M1(binary)
			print "Adding \"{} and {}\" to par lines".format("M2 {} 1".format(M2), "SINI {} 1".format(SINI))
			par_lines.append("\nM2 {} 1\n".format(M2))
			par_lines.append("\nSINI {} 1\n".format(SINI))

			out_dir_file = os.path.join(out_dir, os.path.basename(pulsar))
			if not os.path.exists(out_dir_file):
				print "Making directory {}".format(out_dir_file)
				os.makedirs(out_dir_file)

			to = os.path.join(out_dir_file, os.path.basename(tim_file))
			print "Copying {} --> {}".format(tim_file, to)
			shutil.copy2(tim_file, to)

			out_dir_file = os.path.join(out_dir_file, os.path.basename(par_file))
			print "Writing new par lines to {}".format(out_dir_file)
			out = open(out_dir_file, 'w')
			out.writelines(par_lines)


if __name__ == "__main__":
	main()

