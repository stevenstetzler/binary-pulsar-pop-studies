from psr_constants import Tsun
from psr_constants import SECPERDAY
import numpy as np
import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from glob import glob
import os
import scipy.ndimage.filters as filters
from scipy.interpolate import interp1d
from matplotlib.font_manager import FontProperties
import cPickle as pickle
from PAL2 import bayesutils as bu


def save_object(obj, filename):
    with open(filename, 'wb') as output:
	print "Saving object to {}".format(filename)
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def make_triplot(chains, pars, save_dir=None, visualize=False):
	ax = bu.triplot(chains, labels=pars, tex=False, figsize=(20,15))
	if save_dir is not None:
		save_name = "{}_triplot.png".format("_".join(pars))
		save_file = os.path.join(save_dir, save_name)
		print "Saving triplot to {}".format(save_file)
		plt.savefig(save_file)
	if visualize:
		plt.show()
	plt.close('all')


def make_directory(directory):
        if not os.path.exists(directory):
                print "Making directory {}".format(directory)
                os.makedirs(directory)


def get_par_range(pulsar_dicts, par_name):
        min_par = 1e99
        max_par = -1e99
        for pulsar_dict in pulsar_dicts:
                for pulsar in pulsar_dict:
                        _, _, par_dict = pulsar_dict[pulsar]
                        if par_name in par_dict.keys():
                                par_chain = par_dict[par_name]
                                min_val = par_chain.min()
                                max_val = par_chain.max()
                                if min_val < min_par:
                                        min_par = min_val
                                if max_val > max_par:
                                        max_par = max_val
        return min_par, max_par



def make_plot(par_name, chain, save_dir, description, nbins=80, burn_in=10000, smooth=True, estimate_density=False, bandwidth=None):
	if len(chain) > burn_in:
		chain = chain[burn_in:]

        x_min = chain.min()
        x_max = chain.max()

        plt.figure("single", figsize=(8, 8), dpi=100)
        plt.suptitle(("{} - {}".format(par_name, description)).replace("_", " "))

        ax1 = plt.subplot(211)
        ax1.plot(chain)
        ax1.set_title("{} Trace".format(par_name))
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel(par_name)

        ax2 = plt.subplot(212)
        ax2.hist(chain, nbins, normed=True)
        ax2.set_title("{} Distribution".format(par_name))
        ax2.set_xlabel(par_name)

        if estimate_density:
                X_plot = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
                adjusted_chain = chain[:, np.newaxis]
                if bandwidth is None:
                        for bandwidth in np.linspace(0.0025, 0.05, 8):
                                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                                log_density = kde.score_samples(X_plot)
                                density = np.exp(log_density)
                                density_plot = ax2.plot(X_plot, density, label=str(bandwidth))
                else:
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                        log_density = kde.score_samples(X_plot)
                        density = np.exp(log_density)
                        density_plot = ax2.plot(X_plot, density, label=str(bandwidth))

	vals, binedges = np.histogram(chain, bins=nbins, density=True)
	bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
	vals = filters.gaussian_filter(vals, sigma=0.75)
	f = interp1d(bins, vals, kind='cubic')
	ax2.plot(bins, f(bins), 'r--', lw=3, zorder=2, label="Smoothing")
	ax2.legend()

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.90)

        save_name = os.path.join(save_dir, "{}".format(par_name))
        if estimate_density:
                save_name += "_with_density"

        save_name += ".png"

        print "Saving plot to", save_name
        plt.savefig(save_name)

        plt.close("all")


def make_plot_two_par(par_1, par_1_chain, par_2, par_2_chain, save_dir, description, burn_in=0.1, estimate_density=False, bandwidth=None):
        # Make the chains the same size.
        # There is probably a better way to be doing this, so as to keep the extra data from the longer chain
        # Some kind of interpolation + resizing? scipy.misc.imresize
        if len(par_1_chain) > len(par_2_chain):
                par_1_chain = par_1_chain[:len(par_2_chain),]
        else:
                par_2_chain = par_2_chain[:len(par_1_chain),]

        par_1_chain = par_1_chain[int(burn_in * len(par_1_chain)):]
        par_2_chain = par_2_chain[int(burn_in * len(par_2_chain)):]

        plt.figure("everything", figsize=(12, 12), dpi=100)
        plt.suptitle(("{} vs {} - {}".format(par_1, par_2, description)).replace("_", " "))

        ax1 = plt.subplot(323)
        ax1.plot(par_1_chain)
        ax1.set_title('{} Trace'.format(par_1))
        ax1.set_xlabel('Iteration number')
        ax1.set_ylabel(par_1)

        ax2 = plt.subplot(322)
        ax2.plot(par_2_chain)
        ax2.set_title('{} Trace'.format(par_2))
        ax2.set_xlabel('Iteration number')
        ax2.set_ylabel(par_2)

        ax3 = plt.subplot(325)
        ax3.hist(par_1_chain, 100, normed=True)
        ax3.set_title('{} Distrubution'.format(par_1))
        ax3.set_xlabel(par_1)

        ax4 = plt.subplot(324)
        ax4.hist(par_2_chain, 100, normed=True)
        ax4.set_title('{} Distrubution'.format(par_2))
        ax4.set_xlabel(par_2)

        if estimate_density:
                x_min = par_1_chain.min()
                x_max = par_1_chain.max()

                X_plot = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
                adjusted_chain = par_1_chain[:, np.newaxis]
                if bandwidth is None:
                        for bandwidth in np.linspace(0.0025, 0.05, 8):
                                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                                log_density = kde.score_samples(X_plot)
                                density = np.exp(log_density)
                                density_plot = ax3.plot(X_plot, density, label=str(bandwidth))
                else:
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                        log_density = kde.score_samples(X_plot)
                        density = np.exp(log_density)
                        density_plot = ax3.plot(X_plot, density, label=str(bandwidth))

                ax3.legend()

                x_min = par_2_chain.min()
                x_max = par_2_chain.max()

                X_plot = np.linspace(x_min, x_max, 1000)[:, np.newaxis]
                adjusted_chain = par_2_chain[:, np.newaxis]
                if bandwidth is None:
                        for bandwidth in np.linspace(0.0025, 0.05, 8):
                                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                                log_density = kde.score_samples(X_plot)
                                density = np.exp(log_density)
                                density_plot = ax4.plot(X_plot, density, label=str(bandwidth))
                else:
                        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(adjusted_chain)
                        log_density = kde.score_samples(X_plot)
                        density = np.exp(log_density)
                        density_plot = ax4.plot(X_plot, density, label=str(bandwidth))

                ax4.legend()

        ax5 = plt.subplot(326)
        ax5.hist2d(par_1_chain, par_2_chain, 100)
        ax5.set_title('{} and {} Distrubution'.format(par_1, par_2))
        ax5.set_xlabel(par_1)
        ax5.set_ylabel(par_2)

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.90)

        save_name = os.path.join(save_dir, "{}_vs_{}".format(par_1, par_2))
        if estimate_density:
                save_name += "_with_density"
        save_name += ".png"
        print "Saving plot to", save_name
        plt.savefig(save_name)

        plt.close("all")


def get_par_chain(par_name, chain, pars):
	if par_name == 'PB' and not 'PB' in pars:
		if 'FB0' in pars:
			fb0 = get_par_chain('FB0', chain, pars)
			pb = np.divide(np.divide(1.0, fb0), SECPERDAY)
			return pb 
		else:
			raise RuntimeError("Cannot get PB from chain.")
        if par_name == 'COSI' and not 'COSI' in pars:
                if 'H3' and 'STIG' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        stig = get_par_chain('STIG', chain, pars)
                        sini = 2. * stig / (1. + np.power(stig, 2.))
                        cosi = np.sqrt(1. - np.power(sini, 2.))
                        return cosi
                elif 'H3' and 'H4' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        h4 = get_par_chain('H4', chain, pars)
                        stig = np.divide(h4, h3)
                        sini = 2. * stig / (1. + np.power(stig, 2.))
                        cosi = np.sqrt(1. - np.power(sini, 2.))
                        return cosi
                elif 'SINI' in pars:
                        sini = get_par_chain('SINI', chain, pars)
                        cosi = np.sqrt(1. - np.power(sini, 2.))
                        return cosi
                elif 'KIN' in pars:
                        kin = get_par_chain('KIN', chain, pars)
                        sini = np.sin(kin * np.pi / 180.)
                        cosi = np.sqrt(1. - np.power(sini, 2.))
                        return cosi
                else:
			raise RuntimeError("Cannot get COSI from chain.")
        if par_name == 'M2' and not 'M2' in pars:
                if 'H3' and 'STIG' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        stig = get_par_chain('STIG', chain, pars)
                        shapR = np.divide(h3, np.power(stig, 3.))
                        m2 = shapR / Tsun
                        return m2
                elif 'H3' and 'H4' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        h4 = get_par_chain('H4', chain, pars)
                        stig = np.divide(h4, h3)
                        shapR = np.divide(h3, np.power(stig, 3.))
                        m2 = shapR / Tsun
                        return m2
                else:
			raise RuntimeError("Cannot get M2 from chain.")
	if par_name == 'SINI' and not 'SINI'in pars:
                if 'H3' and 'STIG' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        stig = get_par_chain('STIG', chain, pars)
                        sini = 2. * stig / (1. + np.power(stig, 2.))
                        return sini
                elif 'H3' and 'H4' in pars:
                        h3 = get_par_chain('H3', chain, pars)
                        h4 = get_par_chain('H4', chain, pars)
                        stig = np.divide(h4, h3)
                        sini = 2. * stig / (1. + np.power(stig, 2.))
                        return sini
                elif 'KIN' in pars:
                        kin = get_par_chain('KIN', chain, pars)
                        sini = np.sin(kin * np.pi / 180.)
                        return sini
                else:
			raise RuntimeError("Cannot get SINI from chain.")
	if par_name == 'KIN' and not 'KIN' in pars:
		sini = get_par_chain('SINI', chain, pars)
		return np.arcsin(sini) * 180./np.pi
	if par_name == 'H3' and not 'H3' in pars:
		stig = get_par_chain('STIG', chain, pars)
		m2 = get_par_chain('M2', chain, pars)
		r = Tsun * m2
		h3 = np.multiply(r, np.power(stig, 3.))
		return h3
	if par_name == 'STIG' and not 'STIG' in pars:
		cosi = get_par_chain('COSI', chain, pars)
		stig = np.sqrt(np.divide(1. - cosi, 1 + cosi))
		return stig
	if par_name == 'H4' and not 'H4' in pars:
		h3 = get_par_chain('H3', chain, pars)
		stig = get_par_chain('STIG', chain, pars)
		h4 = np.multiply(h3, stig)
		return h4
	if par_name == 'M1' and not 'M1' in pars:
		m2 = get_par_chain('M2', chain, pars)
		sini = get_par_chain('SINI', chain, pars)
		a1 = get_par_chain('A1', chain, pars)
		pb = get_par_chain('PB', chain, pars) * SECPERDAY

		m1 = np.zeros(len(m2))
		massfunc = (4. * np.pi**2 / Tsun) * np.divide(np.power(a1, 3.), np.power(pb, 2.))
		m1 = np.subtract(np.sqrt(np.divide(np.power(np.multiply(m2, sini), 3.), massfunc)), m2)
#		for ii in range(len(m2)):
#			massfunc = 4 * np.pi**2 * a1[ii]**3 / pb[ii]**2 / Tsun
#			m1[ii] = np.sqrt((m2[ii] * sini[ii])**3 / massfunc) - m2[ii]	
		return m1
		
        par_idx = np.where(par_name == pars)[0][0]
        return chain[:, par_idx]


def get_chain_and_pars(pulsar):
        pulsar_name = os.path.basename(pulsar)

        chain_dir = os.path.join(pulsar, "chains")
        if not os.path.exists(chain_dir):
                print "Pulsar {} did not run (no chains directory found).".format(pulsar_name)
                return None, None
        chain_file = os.path.join(chain_dir, "chain_1.txt")
        if not os.path.exists(chain_file):
                print "No chain file for pulsar {}.".format(pulsar_name)
                return None, None
        chain = np.loadtxt(chain_file)

        par_file = os.path.join(chain_dir, "pars.txt")
        if not os.path.exists(par_file):
                print "No pars for pulsar {}.".format(pulsar_name)
                return None, None
        pars = np.genfromtxt(par_file, dtype='str')

        return chain, pars


def get_pulsar_dict(pulsar_dir):
	pulsars = glob(os.path.join(pulsar_dir, "*"))
  
	pulsar_names = [os.path.basename(p) for p in pulsars]
	for bad_dir in ['part_1', 'part_2', 'broke']:
		if bad_dir in pulsar_names:
			bad_dir_idx = pulsar_names.index(bad_dir)

			replace_dir = pulsars[bad_dir_idx]
			for directory in glob(os.path.join(replace_dir, "*")):
				pulsars.append(directory)
				pulsar_names.append(os.path.basename(directory))

			pulsars.pop(bad_dir_idx)
			pulsar_names.pop(bad_dir_idx)

	print "Processing pulsars:\n", "\t\n".join(pulsar_names)

	ret_dict = {}

	for pulsar in pulsars:
		pulsar_name = os.path.basename(pulsar)
     
		chain, pars = get_chain_and_pars(pulsar)

		if chain is None:
			print "{} - chain missing.".format(pulsar_name)
			continue
		if pars is None:
			print "{} - par file missing.".format(pulsar_name)
			continue

		par_dict = {}
		for par in pars:
#			print "Adding ({}, {}) to dictionary".format(pulsar_name, par)
			par_chain = get_par_chain(par, chain, pars)
			par_dict[par] = par_chain

		if 'COSI' not in pars:
			cosi_chain = get_par_chain("COSI", chain, pars)
			par_dict["COSI"] = cosi_chain

		if 'M2' not in pars:
			m2_chain = get_par_chain("M2", chain, pars)
			par_dict["M2"] = m2_chain				
		
		if 'SINI' not in pars:
			sini_chain = get_par_chain("SINI", chain, pars)
			par_dict["SINI"] = sini_chain

		if 'M1' not in pars:
			m1_chain = get_par_chain("M1", chain, pars)
			par_dict["M1"] = m1_chain	

		if 'KIN' not in pars:
			kin_chain = get_par_chain("KIN", chain, pars)
			par_dict["KIN"] = kin_chain

		if 'PB' not in pars:
			pb_chain = get_par_chain("PB", chain, pars)
			par_dict["PB"] = pb_chain

		if 'H3' not in pars:
			h3_chain = get_par_chain("H3", chain, pars)
			par_dict["H3"] = h3_chain

		if 'STIG' not in pars:
			stig_chain = get_par_chain("STIG", chain, pars)
			par_dict["STIG"] = stig_chain

		ret_dict[pulsar_name] = (chain, pars, par_dict)

	return ret_dict


def get_median_and_bounds(par_name, par_chain, save_name=None, nbins=80, sigma=0.75, visual_check=False):
	print "{0} simulated {1} times.".format(par_name, len(par_chain))
	vals, binedges = np.histogram(par_chain, bins=nbins, density=True)
	bins = binedges[:-1] + (binedges[1] - binedges[0]) / 2
	vals = filters.gaussian_filter(vals, sigma=sigma)
	f = interp1d(bins, vals, kind='cubic')

	pdf = f(bins)
	pdf = np.divide(pdf, np.sum(pdf))

	cdf = [np.sum(pdf[:i]) for i in range(len(pdf))]
	
	one_sigma_lower_bound_prob = 0.1586
	median_prob = 0.5
	one_sigma_upper_bound_prob = 0.8414

	cdf_one_sigma_lower = np.fabs(np.array(cdf) - one_sigma_lower_bound_prob)
	cdf_one_sigma_lower_idx = np.where(cdf_one_sigma_lower == cdf_one_sigma_lower.min())
	par_at_one_sigma_lower_bound = par_chain[cdf_one_sigma_lower_idx][0]
	#print "Found 1 sigma lower bound at bin {0}.".format(cdf_one_sigma_lower_idx[0][0])
	par_at_one_sigma_lower_bound = bins[cdf_one_sigma_lower_idx][0]
	#print par_at_one_sigma_lower_bound

	cdf_median = np.fabs(np.array(cdf) - median_prob)
	cdf_median_idx = np.where(cdf_median == cdf_median.min())
	par_at_median = par_chain[cdf_median_idx][0]
	#print "Found median at bin {0}.".format(cdf_median_idx[0][0])
	par_at_median = bins[cdf_median_idx][0]
	#print par_at_median

	cdf_one_sigma_upper = np.fabs(np.array(cdf) - one_sigma_upper_bound_prob)
	cdf_one_sigma_upper_idx = np.where(cdf_one_sigma_upper == cdf_one_sigma_upper.min())
	par_at_one_sigma_upper_bound = par_chain[cdf_one_sigma_upper_idx][0]
	#print "Found 1 sigma upper bound at bin {0}.".format(cdf_one_sigma_upper_idx[0][0])
	par_at_one_sigma_upper_bound = bins[cdf_one_sigma_upper_idx][0]
	#print par_at_one_sigma_upper_bound

	plt.figure("pdf_and_cdf", figsize=(8, 8), dpi=100)

	ax1 = plt.subplot(211)
	ax1.hist(par_chain, nbins, normed=True, label="Data")
	ax1.plot(bins, f(bins), 'r--', lw=3, zorder=2, label="Smoothed PDF")
	ax1.axvspan(par_at_one_sigma_lower_bound, par_at_median, alpha=0.5, color='red', label="1 Sigma Lower")
	ax1.axvspan(par_at_median, par_at_one_sigma_upper_bound, alpha=0.5, color='yellow', label="1 Sigma Upper")
        ax1.set_title("{0} PDF".format(par_name))
        ax1.set_xlabel("{0} Value".format(par_name))
	ax1.legend()

	ax2 = plt.subplot(212)
	ax2.plot(cdf, 'r--', lw=3, zorder=2, label="CDF")
	ax2.axvspan(cdf_one_sigma_lower_idx[0][0], cdf_median_idx[0][0], alpha=0.5, color='red', label="1 Sigma Lower")
	ax2.axvspan(cdf_median_idx[0][0], cdf_one_sigma_upper_idx[0][0], alpha=0.5, color='yellow', label="1 Sigma Upper")
	ax2.set_title("{0} CDF".format(par_name))
	ax2.set_xlabel("PDF Bin Number")	
	ax2.legend()	

        plt.tight_layout(pad=1.0, w_pad=0.5, h_pad=1.0)
        plt.subplots_adjust(top=0.90)

	if save_name is not None:
		print "Saving {0} PDF and CDF figure to {1}".format(par_name, save_name)
	        plt.savefig(save_name)

	if visual_check:
		plt.show()

	plt.close("all")
	
	return par_at_one_sigma_lower_bound, par_at_median, par_at_one_sigma_upper_bound


def make_autocorrelation_plot(par_name, par_chain, save_name=None, visualize=False):
	vals_unbiased = par_chain - np.mean(par_chain)
	vals_norm = np.sum(np.power(vals_unbiased, 2.))
	auto_corr = np.correlate(vals_unbiased, vals_unbiased, "same") / vals_norm
	auto_corr = auto_corr[len(auto_corr)/2:]
	plt.plot(auto_corr)
	plt.title("Autocorrelation for par {0}".format(par_name))
	
	if save_name is not None:
		print "Saving {} autocorrelation plot to {}".format(par_name, save_name)
		plt.savefig(save_name)
	
	if visualize:
		plt.show()

	plt.close("all")


def make_trace_plot(par_name, par_chain, save_name=None, visualize=False):
	plt.plot(par_chain)
	plt.xlabel("Iteration number")
	plt.ylabel(par_name)
	plt.title("{} Trace".format(par_name))
	
	if save_name is not None:
		print "Saving {} trace plot to {}".format(par_name, save_name)
		plt.savefig(save_name)
	
	if visualize:
		plt.show()

	plt.close("all")


