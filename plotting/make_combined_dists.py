import os
import sys
import cPickle as pickle
from matplotlib import pyplot as plt
import numpy as np
from pylab import savefig


def get_save_filename(par, sim_type):
	plot_save_dir = os.path.join('plots/combined', sim_type)
	plot_save_dir = os.path.join(plot_save_dir, par)
	if not os.path.exists(plot_save_dir):
		os.makedirs(plot_save_dir)
	plot_save_filename = os.path.join(plot_save_dir, "par_trace_and_dist.png")
	return plot_save_filename


def make_plot(par, par_vals, pulsar_chains, sim_type):
	print "Making trace + dist plot for parameter", par

	par_vals = None
	for pulsar_dir, pulsar_par_vals in pulsar_chains:
		print "Concatenating", pulsar_dir.split('/')[len(pulsar_dir.split('/')) - 1]
		if par_vals is None:
			par_vals = pulsar_par_vals
		else:
			par_vals = np.concatenate((par_vals, pulsar_par_vals))
		print par_vals

	trace = plt.subplot(311)
	trace.plot(par_vals)
	trace.set_title(par)
	trace.set_xlabel("Iteration Number")
	trace.set_ylabel(par)
		
	dist = plt.subplot(312)
	weights = np.ones_like(par_vals)/float(len(par_vals))
#	y, x, _ = dist.hist(par_vals, 500, normed=True)
	y, x, _ = dist.hist(par_vals, 500, weights=weights)
	print "Combined max:", y.max()
	dist.set_title(par)
	dist.set_xlabel(par)
	dist.set_ylabel("Frequency")
	
	max_height = 0	
	overlayed = plt.subplot(313)
	for pulsar_dir, pulsar_par_vals in pulsar_chains:
		if pulsar_dir == 'nanograv_11y_data/J1614-2230':
			continue
		weights = np.ones_like(pulsar_par_vals)/float(len(pulsar_par_vals))
#		y, x, _ = overlayed.hist(pulsar_par_vals, 100, normed=True)
		y, x, _ = overlayed.hist(pulsar_par_vals, 100, weights=weights)
		print "Max height:", y.max(), "for", pulsar_dir
		if y.max() > max_height:
			max_height = y.max()
	overlayed.set_ylim([0, max_height * 1.1])
	overlayed.set_title("Overlayed " + par)
	overlayed.set_xlabel(par)
	overlayed.set_ylabel("Frequency")

	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

	filename = get_save_filename(par, sim_type)
	print "Saving trace + dist plot to", filename, "\n"
	savefig(filename)
#	plt.show()	
	plt.close('all')

	

if len(sys.argv) != 2:
	print "Usage:", sys.argv[0], "[pulsar_chains_pickle]"
	exit()

pickle_filename = sys.argv[1]
all_data = pickle.load(open(pickle_filename, 'rb'))

m2sini_par_to_pulsar_chain = all_data[0]
m2sini_par_to_chain = all_data[1]
h3stig_par_to_pulsar_chain = all_data[2]
h3stig_par_to_chain = all_data[3]

print "Making combined plots for parameters", m2sini_par_to_pulsar_chain.keys()

sini_vals = m2sini_par_to_chain['SINI']
sini_pulsars = m2sini_par_to_pulsar_chain['SINI']
try:
	kin_vals = m2sini_par_to_chain['KIN']
	sini_vals = np.concatenate((sini_vals, np.sin(kin_vals * np.pi / 180.)))
	kin_pulsars = m2sini_par_to_pulsar_chain['KIN']
	for pulsar, par_vals in kin_pulsars:
		par_vals = np.sin(par_vals * np.pi / 180.)
		sini_pulsars.append((pulsar, par_vals))
except KeyError:
	pass
cosi_vals = np.sqrt(1. - sini_vals**2)

cosi_pulsars = []
for pulsar, par_vals in sini_pulsars:
	par_vals = np.sqrt(1. - par_vals**2)
	cosi_pulsars.append((pulsar, par_vals))

make_plot('SINI', sini_vals, sini_pulsars, 'm2sini')
make_plot('COSI', cosi_vals, cosi_pulsars, 'm2sini')
#make_plot('KIN', kin_vals, kin_pulsars, 'm2sini')

for par in list(m2sini_par_to_chain.keys()):
	if par == 'SINI' or par == 'KIN':
		continue
	par_vals = m2sini_par_to_chain[par]
	pulsar_chains = m2sini_par_to_pulsar_chain[par]
	make_plot(par, par_vals, pulsar_chains, 'm2sini')

#for par in list(h3stig_par_to_chain.keys()):
#	par_vals = h3stig_par_to_chain[par]
#	pulsar_chains = h3stig_par_to_pulsar_chain[par]
#	make_plot(par, par_vals, pulsar_chains, 'h3stig')

