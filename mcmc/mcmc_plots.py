from matplotlib import pyplot as plt
from pylab import savefig
import numpy as np
import os

def make_plot_two_param(explored_params, param_name_one, param_name_two, save_dir):
	fig = plt.figure("everything", figsize=(10,10))
	param_one_over_all_time = []
	param_two_over_all_time = []
	all_time = []
#	print "There are", len(explored_params), "walkers"
#	print "There are", len(explored_params[0]), "trials"

	for walker in explored_params:
		param_one_over_time = []
		param_two_over_time = []
		time = []
		tic = 0
		for trial in walker:
			for param in trial:
				# print "Testing", param.get_name(), "vs", param_name_one, "and", param_name_two
				if param.get_name() == param_name_one:
					param_one_over_time.append(param.get_val())
					param_one_over_all_time.append(param.get_val())
				if param.get_name() == param_name_two:
					param_two_over_time.append(param.get_val())
					param_two_over_all_time.append(param.get_val())
			time.append(len(time))
			all_time.append(len(all_time))
		# print "Param one", len(param_one_over_time)
	
		plt.figure("everything")
		ax1 = plt.subplot(322)
		ax1.plot(time, param_one_over_time)
		ax1.set_title(param_name_one + ' Trace')
		ax1.set_xlabel('Iteration number')
		ax1.set_ylabel(param_name_one)

		ax2 = plt.subplot(321)
		ax2.plot(time, param_two_over_time)
		ax2.set_title(param_name_two + ' Trace')
		ax2.set_xlabel('Iteration number')
		ax2.set_ylabel(param_name_two)		
	
	plt.figure("everything")

	ax3 = plt.subplot(324)
	param_1_freq, param_1_bins, _ = ax3.hist(np.asarray(param_one_over_all_time, dtype=float))#, 100)
	ax3.set_title(param_name_one + ' Distrubution')
	ax3.set_xlabel(param_name_one)
	
	elem_at_max = np.argmax(param_1_freq)
	param_1_mode = param_1_bins[elem_at_max]

	ax4 = plt.subplot(325)
	param_2_freq, param_2_bins, _ = ax4.hist(np.asarray(param_two_over_all_time, dtype=float))#, 100)
	ax4.set_title(param_name_two + ' Distrubution')
	ax4.set_xlabel(param_name_two)
	
	elem_at_max = np.argmax(param_2_freq)
	param_2_mode = param_2_bins[elem_at_max]

	ax5 = plt.subplot(326)
	ax5.hist2d(np.asarray(param_one_over_all_time, dtype=float), np.asarray(param_two_over_all_time, dtype=float))#, 100)	
	ax5.set_title(param_name_one + ' and ' + param_name_two + ' Distrubution')
	ax5.set_xlabel(param_name_one)
	ax5.set_ylabel(param_name_two)
	
	plt.tight_layout()
	plt.subplots_adjust(hspace=0.3)
#	plt.show()
	num_simulations = len(explored_params)
	num_iterations = len(explored_params[0])
	if save_dir is not None:
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)
		filename = param_name_one + "_" + param_name_two + "_num_simulations_" + str(num_simulations) + "_num_iterations_" + str(num_iterations) + ".png"
		print "Saving figure to " + save_dir + "/" + filename
		savefig(save_dir + "/" + filename)
	plt.close('all')
	return param_1_mode, param_2_mode


def make_plot_one_param(explored_params, param_name, save_dir):
	plt.figure("everything")
	param_over_all_time = []
	all_time = []
	for walker in explored_params:
		param_over_time = []
		time = []
		tic = 0
		for trial in walker:
			for param in trial:
				if param.get_name() is param_name:
					param_over_time.append(param.get_val())
					param_over_all_time.append(param.get_val())
					time.append(len(time))
					all_time.append(len(all_time))
					break
		ax1 = plt.subplot(221)
		ax1.plot(time, param_over_time)
		ax1.set_title(param_name + ' Trace')
		ax1.set_xlabel('Iteration number')
		ax1.set_ylabel(param_name)

		ax2 = plt.subplot(222)
		ax2.hist(param_over_time)#, 100)
		ax2.set_title(param_name + ' Distribution')
		ax2.set_xlabel(param_name)

	ax3 = plt.subplot(223)
	ax3.hist(param_over_all_time)#, 100)
	ax3.set_title(param_name + ' Distrubution')
	ax3.set_xlabel(param_name)

	plt.subplots_adjust(hspace=0.3)
	plt.tight_layout()
#	plt.show()
	num_simulations = len(explored_params)
	num_iterations = len(explored_params[0])
	if not os.exists(save_dir):
		os.makedirs(save_dir)
	filename = param_name + "_num_simulations_" + str(num_simulations) + "_num_iterations_" + str(num_iterations) + ".png"
	print "Saving figure to " + save_dir + "/" + filename
	savefig(save_dir + "/" + filename)
	plt.close('all')


