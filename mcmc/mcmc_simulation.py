import math
import random
from class_defs import parameter
import simulation_utils as su


# Compute an acceptance probability based on the chi-squared fit values of the timing model with the current and proposed parameters
def get_acceptance_prob(chi2_current, chi2_proposed, ndof):
	try:
		prob =  math.exp((-chi2_proposed + chi2_current)/float(ndof))
	except:
		prob = 1.0
	#print "Computed acceptance prob", prob, "from curr:", chi2_current, "and proposed:", chi2_proposed
	return prob

# Get a set of new parameters using the current ones. Assume each paramater comes from a gaussian distribution with a given mean and standard deviation
# Reset the distribution to have a mean of the new value drawn from the previous distrubution
def get_new_params(current_params):
	return_params = []
	for param in current_params:
		if param.get_std_dev() == 0:
			return_params.append(param)
			continue
		#print "Changing param", param.get_name()
		param_mean = param.get_mean()
		param_std_dev = param.get_std_dev()
		# Get a new parameter value by sampling the parameter distribution, a Gaussian is used here
		new_param_val = random.gauss(float(param_mean), float(param_std_dev))
#		new_param_val = random.uniform(float(param_mean) - float(param_std_dev), float(param_mean) + float(param_std_dev))
		return_params.append(parameter(param.get_name(), [new_param_val, new_param_val, param_std_dev]))
	return return_params


# Run the MCMC simulation using the Metropolis algorithm given the intial parameters
# Returns a list of all explored parameters
def run_simulation(initial_params, num_walkers, num_iterations, par_file, tim_file):
	par_read = open(par_file, 'r')
	initial_par_lines = par_read.readlines()

	explored_params = []
	chi2_initial, _, _ = su.run_tempo_with_params(initial_params, par_file, tim_file)
	print "Initial Chi2", chi2_initial
	for walker in range(0, num_walkers):
		print "\nBeginning simulation", walker + 1, "with", num_iterations, "iterations"
		percent_accepted = 0
		walker_explored_params = []
		current_params = initial_params
		chi2_current = chi2_initial
		for iteration in range(0, num_iterations):
			if int(num_iterations/500) != 0:
				if iteration % int(num_iterations/500) == 0:
					print str(100 * (walker/float(num_walkers) +  iteration/float(num_walkers*num_iterations))) + "% done          \r",	
			# Get a new set of parameters, drawing a new mean for the distribution describing each parameter from the curent prior distribution
			proposed_params = get_new_params(current_params)
			# Run tempo with the new set of parameters and grab the chi-squared value of the resulting fit
			chi2_proposed, ndof, _ = su.run_tempo_with_params(proposed_params, par_file, tim_file)
			# print "Proposed Chi2", chi2_proposed
			# A result of None indicates that the fit didn't converge. Do not accept these parameters
			if chi2_proposed is None:
				walker_explored_params.append(current_params)
				continue
			
			# Choose whether or not to keep the proposed parameters based on the proposed and current chi-squared values
			r = random.uniform(0, 1)
			if r < min([get_acceptance_prob(chi2_current, chi2_proposed, ndof), 1]):
				current_params = proposed_params
				chi2_current = chi2_proposed
				percent_accepted += 1.
			walker_explored_params.append(current_params)
		print ""
		explored_params.append(walker_explored_params)
		percent_accepted *= 100./num_iterations
		print "Simulation ", walker + 1, "% Accepted:", percent_accepted
	su.cleanup()
	return  explored_params

