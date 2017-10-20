import sys
import os
from glob import glob
import argparse
from analysis_utils import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("pulsar_dir_1", help="The directory containing tested pulsars")
	parser.add_argument("pulsar_dir_2", help="The second directory containing tested pulsars")
	parser.add_argument("save_dir", help="The location to save plots to")
	parser.add_argument("description", help="A description of the comparison - added to the file name")
	args = parser.parse_args()

	pulsar_dir_1 = args.pulsar_dir_1
	pulsar_dir_2 = args.pulsar_dir_2
	save_dir = args.save_dir
	descr = args.description

	pulsars_1 = glob(os.path.join(pulsar_dir_1, "*"))
	dir_names_1 = [os.path.basename(p) for p in pulsars_1]
	if 'part_1' in dir_names_1:
		replace_dir = pulsars_1[dir_names_1.index('part_1')]
		for directory in glob(os.path.join(replace_dir, "*")):
			pulsars_1.append(directory)
	if 'part_2' in dir_names_1:
		replace_dir = pulsars_1[dir_names_1.index('part_2')]
		for directory in glob(os.path.join(replace_dir, "*")):
			pulsars_1.append(directory)
	
	pulsars_2 = glob(os.path.join(pulsar_dir_2, "*"))
	dir_names_2 = [os.path.basename(p) for p in pulsars_2]
	if 'part_1' in dir_names_2:
		replace_dir = pulsars_2[dir_names_2.index('part_1')]
		for directory in glob(os.path.join(replace_dir, "*")):
			pulsars_2.append(directory)
	if 'part_2' in dir_names_2:
		replace_dir = pulsars_2[dir_names_2.index('part_2')]
		for directory in glob(os.path.join(replace_dir, "*")):
			pulsars_2.append(directory)

	for pulsar_1 in pulsars_1:
		for pulsar_2 in pulsars_2:
			p_name_1 = os.path.basename(pulsar_1)
			p_name_2 = os.path.basename(pulsar_2)

			if p_name_1 == p_name_2:
				out_dir = os.path.join(save_dir, p_name_1)
				if not os.path.exists(out_dir):
					print "Making directory {}".format(out_dir)
					os.makedirs(out_dir)
				chain_1, pars_1 = get_chain_and_pars(pulsar_1)
				chain_2, pars_2 = get_chain_and_pars(pulsar_2)

				if chain_1 is None:
					print "Getting chain and pars failed for pulsar {}".format(pulsar_1)
					continue
				if chain_2 is None:
					print "Getting chain and pars failed for pulsar {}".format(pulsar_2)
					continue				

				cosi_chain_1 = get_par_chain("COSI", chain_1, pars_1)
				cosi_chain_2 = get_par_chain("COSI", chain_2, pars_2)

				make_plot_two_par("COSI_1", cosi_chain_1, "COSI_2", cosi_chain_2, out_dir, descr)		

				m2_chain_1 = get_par_chain("M2", chain_1, pars_1)
				m2_chain_2 = get_par_chain("M2", chain_2, pars_2)
				
				make_plot_two_par("M2_1", m2_chain_1, "M2_2", m2_chain_2, out_dir, descr)

				if 'KIN' in pars_1 and 'KIN' in pars_2:
					kin_chain_1 = get_par_chain("KIN", chain_1, pars_1)
					kin_chain_1 = get_par_chain("KIN", chain_2, pars_2)	
				
					make_plot_two_par("KIN_1", kin_chain_1, "KIN_2", kin_chain_2, out_dir, descr)		

					cosi_chain_1 = np.cos(kin_chain_1)
					cosi_chain_2 = np.cos(kin_chain_2)

					make_plot_two_par("COSI_1", cosi_chain_1, "COSI_2", cosi_chain_2, out_dir, descr)

					if 'M2' in pars_1 and 'M2' in pars_2:
						m2_chain_1 = get_par_chain("M2", chain_1, pars_1)
						m2_chain_2 = get_par_chain("M2", chain_2, pars_2)

						make_plot_two_par("M2_1", m2_chain_1, "M2_2", m2_chain_2, out_dir, descr)

				elif 'SINI' in pars_1 and 'SINI' in pars_2:
					sini_chain_1 = get_par_chain("SINI", chain_1, pars_1)
					sini_chain_2 = get_par_chain("SINI", chain_2, pars_2)

					make_plot_two_par("SINI_1", sini_chain_1, "SINI_2", sini_chain_2, out_dir, descr)

					cosi_chain_1 = np.sqrt(1. - np.power(sini_chain_1, 2.))	
					cosi_chain_2 = np.sqrt(1. - np.power(sini_chain_2, 2.))	

					make_plot_two_par("COSI_1", cosi_chain_1, "COSI_2", cosi_chain_2, out_dir, descr)

					if 'M2' in pars_1 and 'M2' in pars_2:
						m2_chain_1 = get_par_chain("M2", chain_1, pars_1)
						m2_chain_2 = get_par_chain("M2", chain_2, pars_2)

						make_plot_two_par("M2_1", m2_chain_1, "M2_2", m2_chain_2, out_dir, descr)
			
				elif 'H3' in pars_1 and 'H3' in pars_2:
					h3_chain_1 = get_par_chain("H3", chain_1, pars_1)
					h3_chain_2 = get_par_chain("H3", chain_2, pars_2)

					make_plot_two_par("H3_1", h3_chain_1, "H3_2", h3_chain_2, out_dir, descr)

					if 'H4' in pars_1 and 'H4' in pars_2:
						h4_chain_1 = get_par_chain("H4", chain_1, pars_1)
						h4_chain_2 = get_par_chain("H4", chain_2, pars_2)

						make_plot_two_par("H4_1", h4_chain_1, "H4_2", h4_chain_2, out_dir, descr)
			
						stig_chain_1 = np.divide(h4_chain_1, h3_chain_1)
						stig_chain_2 = np.divide(h4_chain_2, h3_chain_2)

						make_plot_two_par("STIG_1", stig_chain_1, "STIG_2", stig_chain_2, out_dir, descr)
					elif 'STIG' in pars_1 and 'STIG' in pars_2:
						stig_chain_1 = get_par_chain("STIG", chain_1, pars_1)
						stig_chain_2 = get_par_chain("STIG", chain_2, pars_2)
					
						make_plot_two_par("STIG_1", stig_chain_1, "STIG_2", stig_chain_2, out_dir, descr)
					else:
						print "Neither H4 nor STIG found in both chains for pulsar {}".format(p_name_1)
						continue
		
#					sini_chain_1 = 2. * stig_chain_1 / (1. + np.power(stig_chain_1, 2.))
#					sini_chain_2 = 2. * stig_chain_2 / (1. + np.power(stig_chain_2, 2.))
#
#					make_plot_two_par("SINI_1", sini_chain_1, "SINI_2", sini_chain_2, out_dir)		
#
#					cosi_chain_1 = np.sqrt(1. - np.power(sini_chain_1, 2.))
#					cosi_chain_2 = np.sqrt(1. - np.power(sini_chain_2, 2.))
#
#					make_plot_two_par("COSI_1", cosi_chain_1, "COSI_2", cosi_chain_2, out_dir)		
#
#					m2_chain_1 = np.divide(h3_chain_1, np.power(stig_chain_1, 3)) / 4.925490947e-6
#					m2_chain_2 = np.divide(h3_chain_2, np.power(stig_chain_2, 3)) / 4.925490947e-6
#
#					make_plot_two_par("M2_1", m2_chain_1, "M2_2", m2_chain_2, out_dir)
				else:
					print "Neither KIN nor SINI nor H3 found in both chains for pulsar {}".format(p_name_1)


if __name__ == '__main__':
	main()
