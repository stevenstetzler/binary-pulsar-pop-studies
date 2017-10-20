import os
import sys
import numpy as np
import glob
import cPickle as pickle
import tempfile
import shutil
import contextlib
import tempo_utils as tu

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


if len(sys.argv) != 2:
	print "Usage:", sys.argv[0], "[data_dir]"
	exit()


def line_is_present(par_name, par_lines):
	par_line = [line for line in par_lines if line.startswith(par_name)]
	if len(par_line) == 1:
		return True
	elif len(par_line) == 0:
		print par_name, "not found in file"
		return False
	else len(par_line) > 1:
		print par_name, "found more than once in file"
		return False


def fit_H3(par_lines, toas):	
	test_lines = par_lines
	BINARY_line = [line for line in test_lines if line.startswith('BINARY')]

	test_lines.remove(BINARY_line)

	BINARY_model = BINARY_line[1]
	if BINARY_model == 'ELL1' or BINARY_model == 'DD':
		BINARY_line = ['BINARY ELL1H']
	test_lines.append(BINARY_line)


	XDOT_present = line_is_present('XDOT', test_lines)
	if not XDOT_present:
		test_lines.append('XDOT 0.0 1')

	for H3 in range(0, 1, 0.00000015):
		test_lines.append('H3 ' + str(H3) + ' 1')
		chi2, ndof, rms, fit_par_lines = tu.run_tempo(toas, test_lines, get_output_par=True)	
		if chi2 is None or chi2/ndof > 2.:
			continue
		else:
			H3_line = [line for line in fit_par_lines if line.startswith('H3')]
			H3_val = H3_line[1]
			if not 0. < H3_val < 1.:
				continue
			else:
				return fit_par_lines
	print "Could not get converging fit for H3"
	return None

def fit_M2(par_lines, toas):
	test_lines = par_lines
	BINARY_line = [line for line in test_lines if line.startswith('BINARY')]

	test_lines.remove(BINARY_line)

	BINARY_model = BINARY_line[1]
	if BINARY_model == 'ELL1H':
		BINARY_line = ['BINARY ELL1']
	test_lines.append(BINARY_line)

	for M2 in range(0, 2, 0.1):
		test_lines.append('M2 ' + str(SINI) + ' 1')
		chi2, ndof, rms, fit_par_lines = tu.run_tempo(toas, test_lines, get_output_par=True)
		if chi2 is None or chi2/ndof > 2.:
			continue
		else:
			M2_line = [line for line in fit_par_lines if line.startswith('M2')]
			M2_val = M2_line[1]
			if not 0. < M2_val:
				continue
			else:
				return fit_par_lines
	print "Could not get converging fit for M2"
	return None


def fit_SINI(par_lines, toas):
	test_lines = par_lines
	BINARY_line = [line for line in test_lines if line.startswith('BINARY')]

	test_lines.remove(BINARY_line)

	BINARY_model = BINARY_line[1]
	if BINARY_model == 'ELL1H':
		BINARY_line = ['BINARY ELL1']
	test_lines.append(BINARY_line)

	for SINI in range(0, 1, 0.1):
		test_lines.append('SINI ' + str(SINI) + ' 1')
		chi2, ndof, rms, fit_par_lines = tu.run_tempo(toas, test_lines, get_output_par=True)
		if chi2 is None or chi2/ndof > 2.:
			continue
		else:
			SINI_line = [line for line in fit_par_lines if line.startswith('SINI')]
			SINI_val = SINI_line[1]
			if not 0. < SINI_val < 1.:
				continue
			else:
				return fit_par_lines
	print "Could not get converging fit for SINI"
	return None

def fit_M2_and_SINI(par_lines, toas):
	


def get_pulsar_name(pulsar):
	return os.path.basename(os.path.normpath(pulsar))

data_dir = sys.argv[1]

pulsar_list = [file for file in glob.glob(os.path.join(data_dir, "*")) if not "." in file and ("J" in file or "B" in file)]

for pulsar in pulsar_list:
	par_file = os.path.join(pulsar, get_pulsar_name(pulsar) + ".init.par")
	if os.path.exists(par_file):
		par_lines = open(par_file, 'r').readlines()

		BINARY_present = line_is_present('BINARY', par_lines)
		M2_present = line_is_present('M2', par_lines)
		SINI_present = line_is_present('SINI', par_lines)
		H3_present = line_is_present('H3', par_lines)
	
		if M2_present and SINI_present:
			# If we have M2 and SINI, then generate H3 + STIG
			m2sini_par_lines = par_lines
			stripped_par_lines = [line for line in par_lines if not line.startswith('M2')]
			stripped_par_lines = [line for line in par_lines if not line.startswith('SINI')]
			H3_par_lines = fit_H3(par_lines)
			H3_par_lines = add_STIG(m2sini_par_lines, H3_par_lines)			

		elif H3_present:
			# If we have H3, generate M2 and SINI, then generate STIG
			stripped_par_lines = [line for line in par_lines if not line.startswith('H3')]
			m2sini_par_lines = fit_M2_and_SINI(stripped_par_lines)
			H3_par_lines = add_STIG(m2sini_par_lines, par_lines)

		elif not M2_present and not SINI_present and not H3_present:
			# If we don't have anything, generate M2 and SINI, then generate H3, then generate STIG
			m2sini_par_lines = fit_M2_and_SINI(par_lines)
			H3_par_lines = fit_H3(par_lines)
			H3_par_lines = add_STIG(m2sini_par_lines, H3_par_lines)

		else:
			print "Strange conditions for pulsar", 	
			continue
	else:
		print "Par file for pulsar", get_pulsar_name(pulsar), "does not exist"
		continue

m2sini_chain_path = 'm2sini/chains/chain_1.txt'
m2sini_par_path = 'm2sini/chains/pars.txt'

h3stig_chain_path = 'h3stig/chains/chain_1.txt'
h3stig_par_path = 'h3stig/chains/pars.txt'

m2sini_par_to_pulsar_chain = {}
m2sini_par_to_chain = {}

h3stig_par_to_pulsar_chain = {}
h3stig_par_to_chain = {}

for pulsar in pulsar_list:
	no_m2sini = False
	no_h3stig = False

	m2sini_chain_file = os.path.join(pulsar, m2sini_chain_path)
	print "Opening:", m2sini_chain_file 

	if os.path.exists(m2sini_chain_file):
		m2sini_chain = np.loadtxt(m2sini_chain_file)
	
		m2sini_par_file = os.path.join(pulsar, m2sini_par_path)

		m2sini_pars = np.genfromtxt(m2sini_par_file, dtype='str')
	else:
		print m2sini_chain_file, "does not exist."
		no_m2sini = True

	h3stig_chain_file = os.path.join(pulsar, h3stig_chain_path)
	print "Opening:", h3stig_chain_file

	if os.path.exists(h3stig_chain_file):
		h3stig_chain = np.loadtxt(h3stig_chain_file)
	
		h3stig_par_file = os.path.join(pulsar, h3stig_par_path)

		h3stig_pars = np.genfromtxt(h3stig_par_file, dtype='str')
	else:
		print h3stig_chain_file, "does not exist."
		no_h3stig = True
	if not no_m2sini:

		for par in m2sini_pars:
			idx = np.where(m2sini_pars == par)[0][0]
			par_data = m2sini_chain[:, idx]
	
			try:
				m2sini_par_to_pulsar_chain[par].append((pulsar, par_data))
				m2sini_par_to_chain[par] = np.concatenate((m2sini_par_to_chain[par], par_data))
			except KeyError:
				m2sini_par_to_pulsar_chain[par] = [(pulsar, par_data)]
				m2sini_par_to_chain[par] = par_data

	if not no_h3stig:
		for par in h3stig_pars:
			if no_h3stig:
				continue
	
			idx = np.where(h3stig_pars == par)[0][0]
			par_data = h3stig_chain[:, idx]

			try:
				h3stig_par_to_pulsar_chain[par].append((pulsar, par_data))
				h3stig_par_to_chain[par] = np.concatenate((h3stig_par_to_chain[par], par_data))
			except KeyError:
				h3stig_par_to_pulsar_chain[par] = [(pulsar, par_data)]
				h3stig_par_to_chain[par] = par_data

chains = [m2sini_par_to_pulsar_chain, m2sini_par_to_chain, h3stig_par_to_pulsar_chain, h3stig_par_to_chain] 
save_object(chains, 'pulsar_chain.pkl')


