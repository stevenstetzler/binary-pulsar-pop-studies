import os
import sys
from glob import glob
import shutil
import argparse
from binary_psr import *
from convert_par_files import line_is_present
from convert_par_files import get_m2_and_sini_from_none_assume_I_M1
from convert_par_files import get_m2_and_sini_from_h3_assume_I_M1
from convert_par_files import get_stig_from_h3_assume_I_M1
from convert_par_files import get_h3_and_stig_from_none_assume_I_M1
from convert_par_files import get_h3_and_stig_from_m2_and_sini
from convert_par_files import get_h3_and_stig_from_m2_and_kin
from math import sin, cos
from math import pi as PI

def get_files_with_par(files, par_name, verbose=False):
	ret_files = []
	for f in files:
		file_lines = open(f, 'r').readlines()
		line = [l for l in file_lines if l.startswith(par_name + ' ')]
		if len(line) == 1:
			ret_files.append(f)
	if verbose:
		print "\nRetrieved files with par {}:\n{}".format(par_name, "\n".join(ret_files))
	return ret_files


def change_ephem_in_pars(par_files, ephem, verbose=False):
	for pf in par_files:
		par_lines = open(pf, 'r').readlines()
		old_ephem_line = [l for l in par_lines if l.startswith('EPHEM ')]
		if len(old_ephem_line) == 1:
			if verbose:
				print "\n{}\n\tChanging {} to {}".format(pf, old_ephem_line[0].split()[1], ephem)
		par_lines = [l for l in par_lines if not l.startswith('EPHEM ')]
		par_lines.append('EPHEM {}'.format(ephem))
		out = open(pf, 'w')
		out.writelines(par_lines)	


def copy_files_to_directory(files, dir_name, verbose=False, pulsar_dirs=True):
	ret_files = []
	for f in files:
		p_name = os.path.basename(f).split("_")[0]

		if pulsar_dirs:
			pulsar_dir = os.path.join(dir_name, p_name)
		else:
			pulsar_dir = dir_name

		make_directory(pulsar_dir, verbose=verbose)

		if os.path.exists(os.path.join(pulsar_dir, "chains")):
			if verbose:
				print "\nChains directory exists for pulsar {}. Not copying files.".format(p_name)
		else:
			file_name = os.path.join(pulsar_dir, os.path.basename(f))
			if verbose:
				print "\nCopying {} --> {}".format(f, file_name)
			shutil.copy2(f, file_name)
			ret_files.append(file_name)
	return ret_files


def get_corresponding_tim_files(pars, tims):
	ret_files = []
	for tim in tims:
		p_name_tim = os.path.basename(tim).split("_")[0]
		for par in pars:
			p_name_par = os.path.basename(par).split("_")[0]
			if p_name_tim == p_name_par:
				ret_files.append(tim)
	return ret_files


def make_directory(dir_name, clean=False, verbose=False):
	if not os.path.exists(dir_name):
		if verbose:
			print "\nMaking directory {0}".format(dir_name)
		os.makedirs(dir_name)
	elif clean:
		rm_response = raw_input("Directory {0} already exists, would you like to remove it? ".format(dir_name))
		while rm_response not in ['y', 'Y', 'yes', 'Yes', 'n', 'N', 'no', 'No']:
			rm_response = raw_input("Directory {0} already exists, would you like to remove it? ".format(dir_name))
		if rm_response in ['y', 'Y', 'yes', 'Yes']:
			shutil.rmtree(dir_name)
			make_directory(dir_name, verbose=verbose)


def change_pars_to_M2_SINI(par_files, verbose=False):
	for par_file in par_files:
		if verbose:
			print "\nProcessing {}".format(par_file)

		binary = binary_psr(par_file)
		pars = dir(binary.par)		
		pulsar_name = binary.par.PSR

		binary_type = binary.par.BINARY
	
		par_lines = open(par_file, 'r').readlines()

		if binary_type in ['ELL1', 'DD']:
			if 'M2' not in pars and 'SINI' not in pars:
				M2, SINI = get_m2_and_sini_from_none_assume_I_M1(binary)
			elif 'M2' in pars and 'SINI' not in pars:
				print "M2 present but SINI is not for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires SINI parameters. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			elif 'M2' not in pars and 'SINI' in pars:
				print "M2 not present but SINI is for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires M2 paramter. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			else:
				continue

			add_par_to_lines(par_lines, "M2", M2)
			add_par_to_lines(par_lines, "SINI", SINI)

		elif binary_type in ['ELL1H', 'DDH']:
			H3_present, H3 = line_is_present(par_lines, 'H3')
			if not H3_present:
				print "Par file {0} requires H3 parameter. Removing from consideration".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			M2, SINI = get_m2_and_sini_from_h3_assume_I_M1(binary, H3)

			remove_par_from_lines(par_lines, 'BINARY')
			remove_par_from_lines(par_lines, 'H3')

			new_binary_type = binary_type.replace("H", "")
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)

			add_par_to_lines(par_lines, "M2", M2)
			add_par_to_lines(par_lines, "SINI", SINI)

			if verbose:
				print "Changing H3 = {0} --> M2 = {1} and SINI = {2}".format(H3, M2, SINI)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)

		elif binary_type in ['DDK']:
#			if verbose:
#				print "Cannot process {0} as {1} is not implemented yet".format(par_file, 'DDK')

			KIN_present, KIN = line_is_present(par_lines, 'KIN')
			if not KIN_present:
				print "Par file {0} requires KIN parameter. Removing from consideration".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			
			SINI = sin(KIN * PI / 180.)

			remove_par_from_lines(par_lines, 'BINARY')
			remove_par_from_lines(par_lines, 'KIN')
			remove_par_from_lines(par_lines, 'KOM')
			remove_par_from_lines(par_lines, 'K96')
			
			new_binary_type = binary_type.replace('K', '')
			
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)
			add_par_to_lines(par_lines, 'SINI', SINI)

			if verbose:
				print "Changing M2 = {0} and KIN = {1} --> M2 = {0} and SINI = {2}".format(M2, KIN, SINI)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)
			
		open(par_file, 'w').writelines(par_lines)	
			

def change_pars_to_H3_H4(par_files, verbose=False):
	for par_file in par_files:
		if verbose:
			print "\nProcessing {}".format(par_file)

		binary = binary_psr(par_file)
		pars = dir(binary.par)		
		pulsar_name = binary.par.PSR

		binary_type = binary.par.BINARY
	
		par_lines = open(par_file, 'r').readlines()

		if binary_type in ['ELL1']:
			if 'M2' not in pars and 'SINI' not in pars:
				M2, SINI = None, None
				H3, STIG = get_h3_and_stig_from_none_assume_I_M1(binary)
			elif 'M2' in pars and 'SINI' not in pars:
				print "M2 present but SINI is not for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires SINI parameters. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			elif 'M2' not in pars and 'SINI' in pars:
				print "M2 not present but SINI is for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires M2 paramter. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			else:
				M2, SINI = binary.par.M2, binary.par.SINI
				H3, STIG = get_h3_and_stig_from_m2_and_sini(binary)

			H4 = H3 * STIG

			# Remove old parameters
			remove_par_from_lines(par_lines, 'BINARY')
			remove_par_from_lines(par_lines, 'M2')
			remove_par_from_lines(par_lines, 'SINI')
			
			# Convert ELL1 --> ELL1H and DD --> DDH
			new_binary_type = binary_type + 'H'
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)

			# Add new parameters
			add_par_to_lines(par_lines, "H3", H3)
			add_par_to_lines(par_lines, "H4", H4)
			
			if verbose:
				print "Changing M2 = {0} and SINI = {1} --> H3 = {2} and H4 = {3}".format(M2, SINI, H3, H4)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)
	
		elif binary_type in ['ELL1H']:
			H3_present, H3 = line_is_present(par_lines, 'H3')
			STIG_present, STIG = line_is_present(par_lines, 'STIG')

			if not H3_present:
				print "Par file {0} requires H3 parameter. Removing from consideration".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue

			if not STIG_present:
				STIG = get_stig_from_h3_assume_I_M1(binary, H3)
			
			H4 = H3 * STIG

			# Add new parameters
			add_par_to_lines(par_lines, "H4", STIG)

			if verbose:			
				print "Changing H3 = {0} --> H3 = {0} and H4 = {1}".format(H3, H4)
		# TODO: Add support for these binary model types - change eccentricity parameterization
		elif binary_type in ['DD', 'DDK', 'DDH']:
			E_present, E = line_is_present(par_lines, 'E')
			OM_present, OM = line_is_present(par_lines, 'OM')
			T0_present, T0 = line_is_present(par_lines, 'T0')
			PB_present, PB = line_is_present(par_lines, 'PB')

			EPS1 = E * sin(OM * PI / 180.)
			EPS2 = E * cos(OM * PI / 180.)
			TASC = T0 - OM / (2 * PI / PB)
			
			if binary_type in ['DDH']:
				H3_present, H3 = line_is_present(par_lines, 'H3')
				if not H3_present:
					print "Par file {0} requires H3 parameter. Removing from consideration".format(par_file)
					os.remove(par_file)
					par_files.remove(par_file)
					continue
				STIG = get_stig_from_h3_assume_I_M1(binary, H3)
			elif binary_type in ['DD']:
				if 'M2' not in pars and 'SINI' not in pars:
					M2, SINI = None, None
					H3, STIG = get_h3_and_stig_from_none_assume_I_M1(binary)
				elif 'M2' in pars and 'SINI' not in pars:
					print "M2 present but SINI is not for pulsar {0}".format(pulsar_name)
					print "Par file {0} requires SINI parameter. Removing from consideration.".format(par_file)
					os.remove(par_file)
					par_files.remove(par_file)
					continue
				elif 'M2' not in pars and 'SINI' in pars:
					print "M2 not present but SINI is for pulsar {0}".format(pulsar_name)
					print "Par file {0} requires M2 paramter. Removing from consideration.".format(par_file)
					os.remove(par_file)
					par_files.remove(par_file)
					continue
				else:
					M2, SINI = binary.par.M2, binary.par.SINI
					H3, STIG = get_h3_and_stig_from_m2_and_sini(binary)
			elif binary_type in ['DDK']:
				KIN_present, KIN = line_is_present(par_lines, 'KIN')
				if not KIN_present:
					print "par file {0} requires KIN parameter. Removing from consideration.".format(par_file)
					os.remove(par_file)
					par_files.remove(par_file)
					continue
				if 'M2' not in pars:
					print "Par file {0} requires M2 paramter. Removing from consideration.".format(par_file)
					os.remove(par_file)
					par_files.remove(par_file)
					continue
				M2 = binary.par.M2
				H3, STIG = get_h3_and_stig_from_m2_and_kin(M2, KIN)	

			H4 = H3 * STIG
				
			# Remove old parameters
			remove_par_from_lines(par_lines, 'BINARY')

			remove_par_from_lines(par_lines, 'E')
			remove_par_from_lines(par_lines, 'OM')
			remove_par_from_lines(par_lines, 'T0')

			remove_par_from_lines(par_lines, 'M2')
			remove_par_from_lines(par_lines, 'SINI')

			remove_par_from_lines(par_lines, 'KIN')
			remove_par_from_lines(par_lines, 'KOM')
			remove_par_from_lines(par_lines, 'K96')
			
			# Convert DD, DDK, DDH --> ELL1H
			new_binary_type = 'ELL1H'
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)

			# Add new parameters
			add_par_to_lines(par_lines, "EPS1", EPS1)
			add_par_to_lines(par_lines, "EPS2", EPS2)
			add_par_to_lines(par_lines, "TASC", TASC)
			
			add_par_to_lines(par_lines, "H3", H3)
			add_par_to_lines(par_lines, "H4", H4)
			
			if verbose:
				print "Changing E = {0}, OM = {1}, and T0 = {2} --> EPS1 = {3}, EPS2 = {4}, and TASC = {5}".format(E, OM, T0, EPS1, EPS2, TASC)
				try:
					print "Changing M2 = {0} and KIN = {1} --> H3 = {2} and H4 = {3}".format(M2, KIN, H3, H4)
				except:
					print "Changing M2 = {0} and SINI = {1} --> H3 = {2} and H4 = {3}".format(M2, SINI, H3, H4)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)

		open(par_file, 'w').writelines(par_lines)	
			
			

def change_pars_to_H3_STIG(par_files, verbose=False):
	for par_file in par_files:
		if verbose:
			print "\nProcessing {}".format(par_file)

		binary = binary_psr(par_file)
		pars = dir(binary.par)		
		pulsar_name = binary.par.PSR

		binary_type = binary.par.BINARY
	
		par_lines = open(par_file, 'r').readlines()

		if binary_type in ['ELL1', 'DD']:
			if 'M2' not in pars and 'SINI' not in pars:
				M2, SINI = None, None
				H3, STIG = get_h3_and_stig_from_none_assume_I_M1(binary)
			elif 'M2' in pars and 'SINI' not in pars:
				print "M2 present but SINI is not for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires SINI parameters. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			elif 'M2' not in pars and 'SINI' in pars:
				print "M2 not present but SINI is for pulsar {0}".format(pulsar_name)
				print "Par file {0} requires M2 paramter. Removing.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			else:
				M2, SINI = binary.par.M2, binary.par.SINI
				H3, STIG = get_h3_and_stig_from_m2_and_sini(binary)

			# Remove old parameters
			remove_par_from_lines(par_lines, 'BINARY')
			remove_par_from_lines(par_lines, 'M2')
			remove_par_from_lines(par_lines, 'SINI')
			
			# Convert ELL1 --> ELL1H and DD --> DDH
			new_binary_type = binary_type + 'H'
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)

			# Add new parameters
			add_par_to_lines(par_lines, "H3", H3)
			add_par_to_lines(par_lines, "STIG", STIG)
			
			if verbose:
				print "Changing M2 = {0} and SINI = {1} --> H3 = {2} and STIG = {3}".format(M2, SINI, H3, STIG)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)

		elif binary_type in ['ELL1H', 'DDH']:
			H3_present, H3 = line_is_present(par_lines, 'H3')
			STIG_present, STIG = line_is_present(par_lines, 'STIG')

			if not H3_present:
				print "Par file {0} requires H3 parameter. Removing from consideration".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue

			if STIG_present:
				continue

			STIG = get_stig_from_h3_assume_I_M1(binary, H3)

			# Add new parameters
			add_par_to_lines(par_lines, "STIG", STIG)

			if verbose:			
				print "Changing H3 = {0} --> H3 = {0} and STIG = {1}".format(H3, STIG)

		elif binary_type in ['DDK']:
			KIN_present, KIN = line_is_present(par_lines, 'KIN')
			if not KIN_present:
				print "par file {0} requires KIN parameter. Removing from consideration.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			if 'M2' not in pars:
				print "Par file {0} requires M2 paramter. Removing from consideration.".format(par_file)
				os.remove(par_file)
				par_files.remove(par_file)
				continue
			M2 = binary.par.M2
			H3, STIG = get_h3_and_stig_from_m2_and_kin(M2, KIN)	

			remove_par_from_lines(par_lines, 'BINARY')
			remove_par_from_lines(par_lines, 'KIN')
			remove_par_from_lines(par_lines, 'KOM')
			remove_par_from_lines(par_lines, 'K96')
			remove_par_from_lines(par_lines, 'M2')
			
			new_binary_type = binary_type.replace('K', 'H')
			
			add_par_to_lines(par_lines, 'BINARY', new_binary_type)
			add_par_to_lines(par_lines, 'H3', H3)
			add_par_to_lines(par_lines, 'STIG', STIG)

			if verbose:
				print "Changing M2 = {0} and KIN = {1} --> H3 = {2} and STIG = {3}".format(M2, KIN, H3, STIG)
				print "Changing {0} --> {1}".format(binary_type, new_binary_type)

		open(par_file, 'w').writelines(par_lines)	
	


def remove_par_from_lines(lines, par_name):
	try:
		par_names = [l.split()[0] for l in lines]
		par_name_idx = par_names.index(par_name)
		lines.pop(par_name_idx)
	except:
		return


def add_par_to_lines(par_lines, par_name, par_val, fit=True):
	if fit:
		fit_val = 1
	else:
		fit_val = 0
	if par_name is not 'BINARY':
		par_lines.append("{0} {1} {2}\n".format(par_name, par_val, fit_val)) 
	else:
		par_lines.append("{0} {1}\n".format(par_name, par_val)) 


def copy_files_with_par(par_files, par_name, path, tim_files=None, convert=None, verbose=False):
	if par_name is None:
		good_par_files = par_files
	# Indexing a string gives a single character
	elif len(par_name[0]) == 1:
		good_par_files = get_files_with_par(par_files, par_name, verbose=verbose)
	# All other param names should be more than 1 character
	else:
		bad_par_files = set(par_files)
		for name in par_name:
			bad_par_files = bad_par_files - set(get_files_with_par(par_files, name))
		good_par_files = list(set(par_files) - bad_par_files)

	good_par_files = copy_files_to_directory(good_par_files, path, verbose=verbose)
	
	if tim_files is not None:
		good_tim_files = get_corresponding_tim_files(good_par_files, tim_files)
		good_tim_files = copy_files_to_directory(good_tim_files, path, verbose=verbose)
	else:
		good_tim_files = tim_files	

	if convert == 'M2_and_SINI':
		change_pars_to_M2_SINI(good_par_files, verbose=verbose)
	elif convert == 'H3_and_STIG':
		change_pars_to_H3_STIG(good_par_files, verbose=verbose)
	elif convert == 'H3_and_H4':
		change_pars_to_H3_H4(good_par_files, verbose=verbose)
	elif convert is not None:
		raise ValueError("Convert keyword '{}' is not accepted".format(convert))

	return good_par_files, good_tim_files


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pardir", dest="par_dir", action='store', type=str, required=True, help="The directory containing all par files for analysis.")
	parser.add_argument("--timdir", dest="tim_dir", action='store', type=str, required=True, help="The directory containing all tim files for analysis.")
	parser.add_argument("--ephem", dest="ephem", action='store', type=str, required=True, help="The ephemeris to change to.")
	parser.add_argument("--outdir", dest="out_dir", action='store', type=str, required=True, help="The directory in which to create the required file structure.")
	parser.add_argument("-v", "--v", action='store_true', help='Verbose mode.')

	args = parser.parse_args()

	par_dir = args.par_dir
	tim_dir = args.tim_dir
	ephem = args.ephem
	out_dir = args.out_dir
	verbose = args.v

	if not os.path.exists(par_dir):
		print "{} does not exist. Please make it and populate with par files".format(par_dir)
		exit()

	if not os.path.exists(tim_dir):
		print "{} does not exist. Please make it and pupulate with tim files".format(tim_dir)
		exit()

	# Get par and tim files
	pars = glob(os.path.join(par_dir, '*.par'))
	tims = glob(os.path.join(tim_dir, '*.tim'))

	# Make a directory for the ephemeris being used
	base_path = os.path.join(out_dir, 'ephem_{0}'.format(ephem))
	make_directory(base_path, clean=True, verbose=verbose)

	# Copy into a new directory the par and tim files
	new_par_dir = os.path.join(base_path, 'par')
	make_directory(new_par_dir, clean=True, verbose=verbose)
	pars = copy_files_to_directory(pars, new_par_dir, verbose=verbose)

	new_tim_dir = os.path.join(base_path, 'tim')
	make_directory(new_tim_dir, clean=True, verbose=verbose)
	tims = copy_files_to_directory(tims, new_tim_dir, verbose=verbose, pulsar_dirs=False)

	# Make a directory for the binary pulsars
	base_path = os.path.join(base_path, 'binary')
	make_directory(base_path, verbose=verbose)

	# Copy all binary pulsars to their own directory in the binary directory
	binary_path = os.path.join(base_path, 'pars_binary')
	binary_pars, _ = copy_files_with_par(pars, 'BINARY', binary_path, verbose=verbose, tim_files=tims)

	# Copy all M2 + SINI pulsars to their own directory
	M2_SINI_unconverted_path = os.path.join(base_path, 'pars_M2_SINI_unconverted')
	M2_SINI_unconverted_pars, _ = copy_files_with_par(binary_pars, ['M2', 'SINI'], M2_SINI_unconverted_path, verbose=verbose, tim_files=tims)

	# Copy all H3 pulsars to their own directory
	H3_path = os.path.join(base_path, 'pars_H3')
	H3_pars, _ = copy_files_with_par(binary_pars, 'H3', H3_path, verbose=verbose, tim_files=tims)

	# Copy and convert all pulsars to M2 + SINI
	M2_SINI_path = os.path.join(base_path, 'pars_M2_SINI')
	M2_SINI_pars, _ = copy_files_with_par(binary_pars, None, M2_SINI_path, convert='M2_and_SINI', verbose=verbose, tim_files=tims)

	# Copy and convert all pulsars to H3 + STIG
	H3_STIG_path = os.path.join(base_path, 'pars_H3_STIG')
	H3_STIG_pars, _ = copy_files_with_par(binary_pars, None, H3_STIG_path, convert='H3_and_STIG', verbose=verbose, tim_files=tims)

	# Copy and convert all pulsars to H3 + H4
	H3_H4_path = os.path.join(base_path, 'pars_H3_H4')
	H3_H4_pars, _ = copy_files_with_par(binary_pars, None, H3_H4_path, convert='H3_and_H4', verbose=verbose, tim_files=tims)

	# Copy pulsars without Shapiro delay detection to their own directory
	none_path = os.path.join(base_path, 'pars_non_detection')
	none_par_names = [os.path.basename(p) for p in binary_pars]

	M2_par_names = [os.path.basename(p) for p in M2_SINI_unconverted_pars]
	M2_set = set(M2_par_names)

	H3_par_names = [os.path.basename(p) for p in H3_pars]
	H3_set = set(H3_par_names)

	binary_par_name = [os.path.basename(p) for p in binary_pars]
	binary_set = set(binary_par_name)

	none_set = binary_set - M2_set - H3_set
	none_par_names = list(none_set)

	none_pars = [p for p in binary_pars if os.path.basename(p) in none_par_names]
	none_pars = copy_files_to_directory(none_pars, none_path, verbose=verbose)

	none_tims = get_corresponding_tim_files(none_pars, tims)
	none_tims = copy_files_to_directory(none_tims, none_path, verbose=verbose)

	for par_files in [none_pars, H3_STIG_pars, H3_H4_pars, M2_SINI_pars, M2_SINI_unconverted_pars, H3_pars, binary_pars]:
		change_ephem_in_pars(par_files, ephem, verbose=verbose)


if __name__ == '__main__':
	main()

