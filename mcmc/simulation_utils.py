from class_defs import parameter
import os
import shutil
import tempo_utils
from decimal import Decimal

def convert_sec_to_hms(position):
	try:
		hours = int(position / 3600.)
	except:
		return position
	position = position - hours * 3600
	minutes = int(position / 60.)
	position = position - minutes * 60
	seconds_whole = str(position)
	seconds = position
	hours = str(hours)
	minutes = str(minutes)
	seconds = str(seconds)

	if len(hours) == 1:
		hours = '0' + hours
	if len(minutes) == 1:
		minutes = '0' + minutes
	if len(seconds_whole) == 1:
		seconds = '0' + seconds

	return hours + ':' + minutes + ':' + seconds


def convert_hms_to_sec(position):
	split_vals = position.split(":")
	try:
		hours = float(split_vals[0])
		minutes = float(split_vals[1])
		seconds = float(split_vals[2])
	except:
		return position
	return 3600.*hours + 60.*minutes + seconds



def get_params_from_par_file(par_filename, requested_pars, all_params=False):
	print "Reading from", par_filename
	par_file = open(par_filename, 'r')
	par_lines = par_file.readlines()
	
	param_par_lines = [line.split() for line in par_lines if (len(line.split()) == 4 and (line.split()[2] == '1' or line.split()[2] == '0'))]

	ret_all_params = []
	for line_item in param_par_lines:
		par_name = line_item[0]
		par_val = line_item[1]
		par_uncert = line_item[3]
		print "Creating parameter:", par_name, [par_val, par_val, par_uncert]

		if 'D' in par_val:
			par_val = par_val.replace('D', 'e')
		if 'D' in par_uncert:
			par_uncert = par_uncert.replace('D', 'e')

		if par_name in ['RAJ', 'DECJ']:
			par_val = convert_hms_to_sec(line_item[1])
			par_uncert = convert_hms_to_sec(line_item[3])
		elif par_name in ['F0', 'F1']:
			par_val = Decimal(line_item[1].replace('D', 'e'))
			par_uncert = Decimal(line_item[3].replace('D', 'e'))
	
		ret_all_params.append(parameter(par_name, [par_val, par_val, par_uncert]))

	ret_req_params = [par for par in ret_all_params if par.get_name() in requested_pars]
			
	return ret_req_params, ret_all_params


def adjust_par_lines(params, par_lines):
	ret_par_lines = []
	for param in params:
		if param.get_std_dev() == 0:
			continue
		ret_par_lines = [line for line in par_lines if not line.startswith(param.get_name())]
	for param in params:
		if param.get_std_dev() == 0:
			continue
		param_val = param.get_val()
		if param.get_name() in ['F1']:
			val_as_str = '{:.12}'.format(param_val)
			if 'e' in val_as_str:
				param_val = val_as_str.replace('e', 'D')
			if 'E' in val_as_str:
				param_val = val_as_str.replace('E', 'D')
			ret_par_lines.append(param.get_name() + ' ' + param_val + ' 0')
		elif param.get_name() in ['RAJ', 'DECJ']:
			param_val = convert_sec_to_hms(param.get_val())
			ret_par_lines.append(param.get_name() + ' ' + param_val + ' 0')
		else:
			ret_par_lines.append(param.get_name() + ' ' + str(param_val) + ' 0')
	return ret_par_lines


def make_par_file_with_params(params, par_file_read, par_file_write):
	par_file = open(par_file_read, 'r')
	par_lines = par_file.readlines()
	for param in params:
		if param.get_std_dev() == 0:
			continue
		par_lines  = [line for line in par_lines if not line.startswith(param.get_name())]

	for param in params:
		if param.get_std_dev() == 0:
			continue
		param_val = param.get_val()
		if param.get_name() in ['F1']:
			val_as_str = '{:.12}'.format(param_val)
			if 'e' in val_as_str:
				param_val = val_as_str.replace('e', 'D')
			if 'E' in val_as_str:
				param_val = val_as_str.replace('E', 'D')
			par_lines.append(param.get_name() + ' ' + param_val + ' 0')
		elif param.get_name() in ['RAJ', 'DECJ']:
			param_val = convert_sec_to_hms(param.get_val())
			par_lines.append(param.get_name() + ' ' + param_val + ' 0')
		else:
			par_lines.append(param.get_name() + ' ' + str(param_val) + ' 0')
	par_out = open(par_file_write, 'w')
	for line in par_lines:
		if '\n' not in line:
			line += '\n'
		par_out.write(line)


def cleanup():
	for root, dirs, files in os.walk('temp'):
		for f in files:
			os.unlink(os.path.join(root, f))
		for d in dirs:
			shutil.rmtree(os.path.join(root, d))	
	os.rmdir('temp')


# Run Tempo with the given parameters, using the tempo_utils library
# Returns the resulting chi-squared fit value for the model using the given parameters
def run_tempo_with_params(params, par_file, tim_file):
	#for par in params:
		#print "Running Tempo with par", par.get_name(),"=", str(par.get_val())

	if not os.path.exists('temp'):
		os.mkdir('temp')

	temp_par_filename = 'temp_par.par'

	make_par_file_with_params(params, par_file, temp_par_filename)

#	new_par_lines = adjust_par_lines(params, par_lines)
#	print "Running tempo with params:"
#	for p in new_par_lines:
#		print p,
	
	toas = tempo_utils.read_toa_file(tim_file, ignore_blanks=True, convert_skips=True)
	(chi_squared, num_doff, rms) = tempo_utils.run_tempo(toas, temp_par_filename)	

	return chi_squared, num_doff, rms


def get_print_param_filename(params, num_walkers, num_iterations, date, everything=False):
	param_names = []
	for param in params:
		param_names.append(param.get_name())
	param_names = sorted(param_names)
	param_names_print = ''
	for name in param_names:
		param_names_print += name + "_"	
	dir_path = "simulation_results/" + date
	if everything:
		dir_path = "simulation_results/all/" + date
		param_names_print = 'all_params_'
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
        filename = dir_path + "/" + param_names_print + "num_walkers_" + str(num_walkers) + "_num_iterations_" + str(num_iterations) + ".txt"
	return filename


def print_params_to_file(explored_params, filename):
	print "Printing MCMC results to " + filename
	out = open(filename, 'w')
	for walker in explored_params:
		out.write("New simulation\n")
		for trial in walker:
			out_string = ''
			for param in trial:
				out_string += str(param.get_name()) + ":" + str(param.get_val()) + '\t'
			out_string += '\n'
			out.write(out_string)
	out.close()

