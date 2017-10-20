import math
import random
import tempo_utils
from decimal import Decimal
import numpy as np
import mcmc_plots as mp
from class_defs import parameter
import simulation_utils as su
import mcmc_simulation
import sys


def parse_file_for_data(filename):
	in_file = open(filename, 'r')
	file_lines = in_file.readlines()
	ret_data = []
	num_simulations = 0
	for line in file_lines:
		if line == 'New simulation\n':
			num_simulations += 1
			print 'Num Simulations:', num_simulations
	for i in range(num_simulations):
		ret_data.append([])

	curr_simulation = -1
	if num_simulations == 0:
		ret_data.append([])
		curr_simulation = 0
	for line in file_lines:
		if line == 'New simulation\n':
			curr_simulation += 1
			continue

		data_split = line.split()
		if len(data_split) == 1:
			continue
		params = []
		for data in data_split:
			name_and_val = data.split(":")
			name = name_and_val[0]
			val = name_and_val[1]	
			if name in ['F0', 'F1']:
				val = Decimal(val)
			else:
				val = float(val)
			curr_param = parameter(name, [val, val, 0.])
			params.append(curr_param)
		ret_data[curr_simulation].append(params)
	return ret_data


def main():
	if len(sys.argv) == 1:
		print "Usage:", sys.argv[0], "[filename]"
		exit()

	filename = sys.argv[1]

	dirs = filename.split("/")
	date = dirs[len(dirs) - 2]
	save_dir = "plots"
	for i in range(1, len(dirs) - 1):
		save_dir += "/" + dirs[i]

	all_data = parse_file_for_data(filename)

	first_iter = all_data[0][0]
	params_to_test = []
	for param in first_iter:
		params_to_test.append(param.get_name())
	print "Making plots with parameters", params_to_test

	burn_in = 0 # int(0.10 * len(all_data[0]))

	good_data = [data[burn_in:] for data in all_data]


	i = 1
	for param_one in params_to_test:
		for param_two in params_to_test[i:]:
			if param_two == param_one:
				continue
			param_one_mode, param_two_mode = mp.make_plot_two_param(good_data, param_one, param_two, "plots/all/" + date)
			print param_one, "mode:", str(param_one_mode), "\n" + param_two, "mode:", str(param_two_mode)
		i += 1


if __name__ == '__main__':
	main()

