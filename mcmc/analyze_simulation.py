import sys
import os
from simulation_utils import convert_sec_to_hms

if len(sys.argv) < 5:
	print "Usage: python " + sys.argv[0] + " [all - optional] [param_1] [param_2] [date] [number_of_simulations] [number_of_iterations]"
	print "EX: python " + sys.argv[0] + " F0 DECJ 05_31_2017 4 40000"
	print "EX: python " + sys.argv[0] + " all 05_31_2017 4 40000"
	exit()
if len(sys.argv) == 5:
	if sys.argv[1] == 'all':
		run_all = True
	else:
		run_all = False
	date = sys.argv[2]
	num_sims = sys.argv[3]
	num_iter = sys.argv[4]
	print "Running with all parameters not implemented."
	exit()	
elif len(sys.argv) == 6:
	param_1 = sys.argv[1]
	param_2 = sys.argv[2]
	date = sys.argv[3]
	num_sims = sys.argv[4]
	num_iter = sys.argv[5]

acceptable_params = ['F0', 'F1', 'DECJ', 'RAJ', 'DM']

if param_1 not in acceptable_params:
	print "Parameter " + param_1 + " not usable."
	print "The list of acceptable parameters are", acceptable_params
if param_2 not in acceptable_params:
	print "Parameter " + param_2 + " not usable."
	print "The list of acceptable parameters are", acceptable_params

file_name = "simulation_results/" + date + "/" + param_1 + "_" + param_2 + "_num_walkers_" + num_sims + "_num_iterations_" + num_iter + ".txt"
file_name_switch = "simulation_results/" + date + "/" + param_2 + "_" + param_1 + "_num_walkers_" + num_sims + "_num_iterations_" + num_iter + ".txt"

if os.path.exists(file_name):
	print "Opening " + file_name
	in_file = open(file_name, 'r')
elif os.path.exists(file_name_switch):
	print "Opening " + file_name_switch
	in_file = open(file_name_switch, 'r')
else:
	print "No simulation was run with parameters " + param_1 + " and " + param_2
	exit()

lines = in_file.readlines()
avg_f0 = 0.
count = float(len(lines))
accumulator = []
accumulator.append([param_1])
accumulator.append([param_2])
for line in lines:
	if line == 'New Simulation\n':
		continue
	data = line.split()
	for datum in data:
		sp = datum.split(":")
		if sp[0] == accumulator[0][0]:
			accumulator[0].append(float(sp[1]))
		if sp[0] == accumulator[1][0]:
			accumulator[1].append(float(sp[1]))
avg_param_1 = sum(accumulator[0][1:])/len(accumulator[0][1:])
avg_param_2 = sum(accumulator[1][1:])/len(accumulator[1][1:])
if param_1 == 'DECJ' or param_1 == 'RAJ':
	avg_param_1 = convert_sec_to_hms(avg_param_1)
if param_2 == 'DECJ' or param_2 == 'RAJ':
	avg_param_2 = convert_sec_to_hms(avg_param_2)

print "Avg " + param_1 + ":", avg_param_1
print "Avg " + param_2 + ":", avg_param_2
