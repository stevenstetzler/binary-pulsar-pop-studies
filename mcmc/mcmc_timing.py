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


# TODO: Replace with python argparser
if len(sys.argv) == 1:
	print "Proper usage:", sys.argv[0], "[params_to_test] [num_simulations] [num_iterations] [save_directory] [par_file] [tim_file]"
	print "e.g.", sys.argv[0], "RAJ DECJ FO", "3", "1000", "05_31_2017", "B1855+09.par", "B1855+09.tim"
	exit()

params_to_test = [sys.argv[i] for i in range(1, len(sys.argv) - 5)]
num_walkers = int(sys.argv[len(sys.argv) - 5])
num_iterations = int(sys.argv[len(sys.argv) - 4])
burn_in = int(num_iterations/10)
save_dir = sys.argv[len(sys.argv) - 3]
par_file = sys.argv[len(sys.argv) - 2]
tim_file = sys.argv[len(sys.argv) - 1]

if 'all' in params_to_test:
	test_all_pars = True
	params_to_test.remove('all')
else:
	test_all_pars = False

requested_params, all_params = su.get_params_from_par_file(par_file, params_to_test)

for p in all_params:
	if p.get_name() in ['F0', 'F1']:
		p = parameter(p.get_name(), [p.get_val(), p.get_val(), Decimal(2.0) * p.get_std_dev()])
for p in requested_params:
	if p.get_name() in ['F0', 'F1']:
		p = parameter(p.get_name(), [p.get_val(), p.get_val(), Decimal(2.0) * p.get_std_dev()])

if test_all_pars:
	all_param_names = [par.get_name() for par in all_params]

	print "\nRunning MCMC with all parameters"
	for param in all_params:
		print param.get_name(), "=", param.get_val(), "+-", param.get_std_dev()

	explored_params = mcmc_simulation.run_simulation(all_params, num_walkers, num_iterations, par_file, tim_file)

	good_data = [data[burn_in:] for data in explored_params]

	filename = su.get_print_param_filename(all_params, num_walkers, num_iterations, save_dir, everything=True)
	su.print_params_to_file(good_data, filename)

	i = 1
	for param_one in all_params:
		for param_two in all_params[i:]:
			if param_two.get_name() == param_one.get_name():
				continue
			mp.make_plot_two_param(good_data, param_one.get_name(), param_two.get_name(), "plots/all/" + save_dir)
		i += 1

# If user only specified all
if len(params_to_test) == 0:
	exit()

if len(params_to_test) == 1:
	print "Need to specific more than one parameter to test pairwise"
	exit()

i = 1
for param_one in requested_params:
	for param_two in requested_params[i:]:
		if param_two.get_name() == param_one.get_name():
			continue
		run = []
		for param in requested_params:
			if param.get_name() != param_one.get_name() and param.get_name() != param_two.get_name():
				run.append(parameter(param.get_name(), [param.get_val(), param.get_mean(), 0]))
			else:
				run.append(parameter(param.get_name(), [param.get_val(), param.get_mean(), param.get_std_dev()]))	
		print "\nRunning simulation with", param_one.get_name(), "and", param_two.get_name()
		for param in run:
			print param.get_name(), "=", param.get_val(), "+-", param.get_std_dev()

		explored_params = mcmc_simulation.run_simulation(run, num_walkers, num_iterations, par_file, tim_file)

		filename = su.get_print_param_filename([p for p in run if p.get_std_dev() != 0], num_walkers, num_iterations, save_dir)
		good_data = [data[burn_in:] for data in explored_params]

	        su.print_params_to_file(good_data, filename)
		mp.make_plot_two_param(good_data, param_one.get_name(), param_two.get_name(), "plots/" + save_dir)

	i += 1


# Define the initial parameter values to use

#RA = parameter('RAJ', [su.convert_hms_to_sec('01:58:46.00091762'), su.convert_hms_to_sec('01:58:46.00091762'), 1. * su.convert_hms_to_sec('0:0:0.00672614')])
#RA = parameter('RAJ', [su.convert_hms_to_sec('01:58:46.00091762'), su.convert_hms_to_sec('01:58:46.00091762'), su.convert_hms_to_sec('0:0:0.24072614')])

#DEC = parameter('DECJ', [su.convert_hms_to_sec('21:06:46.5552421'), su.convert_hms_to_sec('21:06:46.5552421'), 1. * su.convert_hms_to_sec('0:0:0.3137359')])
#DEC = parameter('DECJ', [su.convert_hms_to_sec('21:06:46.5552421'), su.convert_hms_to_sec('21:06:46.5552421'), su.convert_hms_to_sec('0:0:2.5137359')])

#F0 = parameter('F0', [Decimal('1.97903061264'), Decimal('1.97903061964'), Decimal('5.09179066553e-10')])
#F0 = parameter('F0', [Decimal('1.9790306196588379'), Decimal('1.9790306196588379'), Decimal('2.') * Decimal('1.0814678e-11')])
			       
#F1 = parameter('F1', [Decimal('-1.382927057796e-15'), Decimal('-1.382927057796e-15'), Decimal('2.') * Decimal('2.893039923855e-18')])

#DM = parameter('DM', [19.869791, 19.869791, 1. * 0.009364])
#DM = parameter('DM', [19.869791, 19.869791, 0.302364])

#params = [RA, DEC, F0, F1, DM]

# Run an MCMC simulation with all parameters being sampled at once. Produce plots for each pair


# Continue simulations with each pair of other specified parameters
# Set standard deviation of all parameters we don't wish to test to 0
#if 'RAJ' not in params_to_test:
#	RA.set_pars([RA.get_val(), RA.get_mean(), 0])
#
#if 'DECJ' not in params_to_test:
#	DEC.set_pars([DEC.get_val(), DEC.get_mean(), 0])
#
#if 'F0' not in params_to_test:
#	F0.set_pars([F0.get_val(), F0.get_mean(), 0])
#
#if 'F1' not in params_to_test:
#	F1.set_pars([F1.get_val(), F1.get_mean(), 0])
#
#if 'DM' not in params_to_test:
#	DM.set_pars([DM.get_val(), DM.get_mean(), 0])

# Run an MCMC simulation for each pair of parameters, avoiding duplicate pairs
