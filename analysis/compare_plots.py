from subprocess import call
import os
import argparse
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir_1", help="The first directory to look for pulsar plots in.", required=True)
	parser.add_argument("--dir_2", help="The second dreictory to look for pulsar plots in.")
	parser.add_argument("--pars", nargs='+', help="The parameters to preview plots of.", required=True)
	
	args = parser.parse_args()

	dir_1 = args.dir_1
	dir_2 = args.dir_2
	pars = args.pars

	pulsars_1 = glob(os.path.join(dir_1, "*"))
	pulsars_1 = [p for p in pulsars_1 if ('J' in os.path.basename(p) or 'B' in os.path.basename(p)) and ('+' in os.path.basename(p) or '-' in os.path.basename(p))]
	
	if dir_2 is not None:
		print "Comparing pulsar parameters from two different simulations."

		pulsars_2 = glob(os.path.join(dir_2, "*"))
		pulsars_2 = [p for p in pulsars_2 if ('J' in os.path.basename(p) or 'B' in os.path.basename(p)) and ('+' in os.path.basename(p) or '-' in os.path.basename(p))]
		
		for pulsar_1 in pulsars_1:
			for pulsar_2 in pulsars_2:
				p_1_name = os.path.basename(pulsar_1)
				p_2_name = os.path.basename(pulsar_2)
				if not p_1_name == p_2_name:
					continue
				for par in pars:
					par_image_name_1 = os.path.join(pulsar_1, "{}_{}.png".format(p_1_name, par.upper()))
					par_image_name_2 = os.path.join(pulsar_2, "{}_{}.png".format(p_2_name, par.upper()))
					if os.path.exists(par_image_name_1) and os.path.exists(par_image_name_2):
						image_1 = mpimg.imread(par_image_name_1)
						image_2 = mpimg.imread(par_image_name_2)
						fig, axes = plt.subplots(1, 2, figsize=(10, 10))
						ax1 = axes[0]
						ax2 = axes[1]
						ax1.set_title(" ".join(pulsar_1.split("/")[-2:]))
						ax2.set_title(" ".join(pulsar_2.split("/")[-2:]))
						ax1.imshow(image_1)
						ax2.imshow(image_2)
						plt.tight_layout()
						plt.draw()
						plt.waitforbuttonpress(0)
						plt.close(fig)
					else:
						print "{} exists: {}".format(par_image_name_1, os.path.exists(par_image_name_1))
						print "{} exists: {}".format(par_image_name_2, os.path.exists(par_image_name_2))
	else:
		print "Comparing parameters for one simulation."
		dir_2 = dir_1
		
		if len(pars) <= 1:
			print "Must pass more than one parameter to observe."
			exit()

		for pulsar in pulsars_1:
			pulsar_name = os.path.basename(pulsar)
			for i, par_1 in enumerate(pars):
				for par_2 in pars[i+1:]:
					par_image_name_1 = os.path.join(pulsar, "{}_{}.png".format(pulsar_name, par_1.upper()))
					par_image_name_2 = os.path.join(pulsar, "{}_{}.png".format(pulsar_name, par_2.upper()))
					if os.path.exists(par_image_name_1) and os.path.exists(par_image_name_2):
						image_1 = mpimg.imread(par_image_name_1)
						image_2 = mpimg.imread(par_image_name_2)
						fig, axes = plt.subplots(1, 2, figsize=(10, 10))
						ax1 = axes[0]
						ax2 = axes[1]
						ax1.set_title(" ".join(pulsar.split("/")[-2:]))
						ax2.set_title(" ".join(pulsar.split("/")[-2:]))
						ax1.imshow(image_1)
						ax2.imshow(image_2)
						plt.tight_layout()
						plt.draw()
						plt.waitforbuttonpress(0)
						plt.close(fig)
					else:
						print "{} exists: {}".format(par_image_name_1, os.path.exists(par_image_name_1))
						print "{} exists: {}".format(par_image_name_2, os.path.exists(par_image_name_2))


if __name__ == "__main__":
	main()
