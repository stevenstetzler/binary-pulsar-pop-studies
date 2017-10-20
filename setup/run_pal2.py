import os
from glob import glob
import sys
from subprocess import call
from subprocess import Popen
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pardir", required=True, help="The directory containing individual pulsar directories with par files. This should be teh directory pars_* as output by organize_files.py")
	parser.add_argument("--niter", type=str, default='10000000', help="The number of iterations to run in each MCMC simluation.")
	parser.add_argument("--neff", type=str, default='1000', help="The number of effective samples to run simulations for. Effective sample size is unique to each simulation.")
	parser.add_argument("--thin", type=str, default='1', help="The thinning factor to use when printing out the chain. PAL2 uses a default of 10, but this script's default is 1.")
	parser.add_argument("--timdir", default='par', help="The directory containing all tim files for the pulsars. The default value 'par' assumes that TOAs are stored in the pulsar directories in the same location as the par directory.")
	parser.add_argument("--here", action="store_true", help="Use the use the shell for stdout. Otherwise a file called \"simulation_out.txt\" will be used.")

	args = parser.parse_args()
	
	par_dir = args.pardir
	tim_dir = args.timdir
	niter = args.niter
	neff = args.neff
	thin = args.thin
	here = args.here

	if int(neff) >= 1000 and int(niter) <= 1e6:
		print "WARNING: The number of iterations {} might not be sufficient to have {} effective samples.".format(niter, neff)

	pulsar_dirs = glob(os.path.join(par_dir, "*"))
	for pulsar_dir in pulsar_dirs:
		pulsar_name = os.path.basename(pulsar_dir)		
#		print "Pulsar {}".format(pulsar_name)
#		print "Pulsar dir {}".format(pulsar_dir)
		par_dir = pulsar_dir
		if tim_dir == 'par':
			psr_tim_dir = pulsar_dir

#		print "Checking {} for par files.".format(par_dir)
		pars = glob(os.path.join(par_dir, "*.par"))
		pars_exist = len(pars) > 0
#		print "Found\n{}".format("\n\t\t\n".join(pars))

		tims = glob(os.path.join(psr_tim_dir, "*.tim"))
#		print "\n".join([os.path.basename(t) for t in tims])
		tim_file = [t for t in tims if os.path.basename(t).startswith(pulsar_name)]
#		print "Found {}".format("\n".join(tim_file))
		tims_exist = len(tim_file) > 0

		hdf5_exists = len(glob(os.path.join(pulsar_dir, "{}.hdf5".format(pulsar_name)))) > 0

		out_filename = os.path.join(pulsar_dir, "simulation_out.txt")
		out_file = open(out_filename, 'a')		
	
		out_dir = os.path.join(pulsar_dir, "chains")

		hdf5_file = os.path.join(pulsar_dir, "{}.hdf5".format(pulsar_name))

		if not hdf5_exists and pars_exist and tims_exist:
			tim_file = tim_file[0]
			print "Making hdf5 file for pulsar {}".format(pulsar_name)
			command = ['makeH5file.py', '--h5File', hdf5_file, '--pardir', par_dir, '--timdir', psr_tim_dir]
			print " ".join(command)			
			if here:
				call(command)
			else:
				call(command, stdout=out_file)

		hdf5_exists = len(glob(os.path.join(pulsar_dir, "{}.hdf5".format(pulsar_name)))) > 0

		if not os.path.exists(out_dir):

			if pars_exist and tims_exist and hdf5_exists:
				print "Running PAL2 timing model for pulsar {}".format(pulsar_name)
				command = ['PAL2_run.py', '--h5File', hdf5_file, '--pulsar', pulsar_name, '--outDir', out_dir, '--mark9', '--incEquad', '--incJitterEquad', '--incRed', '--niter', niter, '--neff', neff, '--thin', thin, '--incTimingModel', '--tmmodel', 'nonlinear'] 
				print " ".join(command)
				if here:
					call(command)
				else:
					call(command, stdout=out_file)
#				Popen(command)
			else:
				print "hdf5 file or par files or tim files missing for pulsar {}".format(pulsar_name)
		else:
			print "{} already ran".format(pulsar_name)
		print ""
		out_file.close()

if __name__ == "__main__":
	main()

