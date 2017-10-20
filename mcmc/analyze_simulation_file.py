import sys
import os
from simulation_utils import convert_sec_to_hms
from simulation_utils import make_par_file_with_params
from math import sqrt
from decimal import Decimal
from class_defs import parameter
from matplotlib import pyplot as plt
import numpy as np


def main():
	if len(sys.argv) != 3:
		print "Usage: python " + sys.argv[0] + " [filename] + [par_file]"
		exit()

	filename = sys.argv[1]
	par_file = sys.argv[2]
	if not os.path.exists(filename):
		print "File " + filename + "does not exists"
		exit()

	in_file = open(filename, 'r')

	acceptable_params = ['F0', 'F1', 'DECJ', 'RAJ', 'DM']
	lines = in_file.readlines()
	accumulator = []
	for param in acceptable_params:
		if param in ['F0', 'F1']:
			accumulator.append([param, Decimal(0.), Decimal(0.), []])
		else:
			accumulator.append([param, 0., 0., []])

	counts = float(len(lines))

	for line in lines:
		if line == 'New simulation\n':
			continue	
		iteration_data = line.split()
		for param_vals in iteration_data:
			name_and_data = param_vals.split(":")
			for param in accumulator:
				param_name = name_and_data[0]
				if param_name in ['F0', 'F1']:
					param_val = Decimal(name_and_data[1])
				else:
					param_val = float(name_and_data[1])
				if param_name == param[0]:
					param[1] += param_val
					param[2] += param_val * param_val
					param[3].append(param_val)
	for param in accumulator:
		if param[0] in ['F0', 'F1']:
			counts = Decimal(counts)
		else:
			counts = float(counts)
		# The average
		param[1] /= counts
		#print "E[x]^2: " + str(param[1] * param[1])
		#print "E[x^2]: " + str(param[2] / counts)
		# The standard deviation
		# Definition: Variance(x) = E[x^2] - E[x]^2 = Std. Dev.^2
		param[2] = sqrt(param[2] / counts - param[1] * param[1])
		if param[0] == 'RAJ' or param[0] == 'DECJ':
			param[1] = convert_sec_to_hms(param[1])
			param[2] = convert_sec_to_hms(param[2])
		freq, bins, _ = plt.hist(np.asarray(param[3], dtype='float'))
		elem_at_max = np.argmax(freq)
		param_mode = bins[elem_at_max]
		param[3] = param_mode
		print str(param[0]), "Avg:", str(param[1]), "Std. Dev.:", str(param[2]), "Mode:", str(param_mode)

	# Get rid of leading directories
	fn_split = filename.split("/")
	fn_ext = fn_split[len(fn_split) - 1]
	# Replace extension with .par
	fn_ext_split = fn_ext.split(".")
	fn_averages = "testing_par_files/" + fn_ext_split[0] + "_averages.par"
	fn_modes = "testing_par_files/" + fn_ext_split[0] + "_modes.par"

	if not os.path.exists("testing_par_files"):
		os.makedirs("testing_par_files")

	print "\nPrinting results to " +  fn +  " for testing with tempo"
	params_to_print = []
	for param in accumulator:
		params_to_print.append(parameter(param[0], [param[1], param[1], param[2]]))
	make_par_file_with_params(params_to_print, par_file, fn_averages)
	param_to_print = []
	for param in accumulator:
		params_to_print.append(parameter(param[0], [param[3], param[3], param[2]]))
	make_par_file_with_params(params_to_print, par_file, fn_modes)

if __name__ == '__main__':
	main()

