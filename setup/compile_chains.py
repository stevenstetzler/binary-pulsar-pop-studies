from analysis_utils import save_object, get_pulsar_dict
import argparse
from organize_files import make_directory
import os
import numpy as np

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--simulation_dirs", nargs='+', help="A list of directories containing all pulsars with chains to be analyzed", required=True)
	parser.add_argument("--save_dir", help="The name to save the compiled chains to.", required=True)
	parser.add_argument("--pars", nargs='+', default=[], help="The parameters to compile chains for.", required=False)
	
	args = parser.parse_args()

	sim_dirs = args.simulation_dirs
	save_dir = args.save_dir
	pars = args.pars	

	make_directory(save_dir, clean=True, verbose=True)

	pulsar_dicts = []
	chain = None


	for sim_dir in sim_dirs:
		pulsar_dict = get_pulsar_dict(sim_dir)
		pulsar_dicts.append(pulsar_dict)
		simulation_type = os.path.dirname(sim_dir).split("/")[-1]
		save_object((pulsar_dict, simulation_type), os.path.join(save_dir, "{}_pulsar_chain_dict.pkl".format(simulation_type)))		

	for par in pars:
		chains = []
		weights = []
		chain_length = 0
		for sim_dir, pulsar_dict in zip(sim_dirs, pulsar_dicts):
			simulation_type = os.path.dirname(sim_dir).split("/")[-1]
			chains_sim = []
			weights_sim = []
			for pulsar in pulsar_dict:
				_, _, par_dict = pulsar_dict[pulsar]

				chain = par_dict[par]
				chains.append(chain)
				chains_sim.append(chain)

				weights_chain = np.ones_like(chain).astype(float) / len(chain)
				weights.append(weights_chain)
				weights_sim.append(weights_chain)

				if chain is not None:
					chain = np.concatenate((chain, chain))
				else:
					chain = chain

			weights_sim = [weight * (len(chain) - chain_length) for weight in weights_sim]
			chain_length = len(chain)

			save_object((chains_sim, weights_sim), os.path.join(save_dir, "{}_{}_chains_with_weights.pkl".format(simulation_type, par)))
				
		weights = [weight * len(chain) for weight in weights]

		save_object((chains, weights), os.path.join(save_dir, "{}_chains_with_weights.pkl".format(par)))
		save_object(chain, os.path.join(save_dir, "{}_chain.pkl".format(par)))

#	final_pulsar_dict = {}
#	for pulsar_dict in pulsar_dicts:
#		for pulsar in pulsar_dict:
#			final_pulsar_dict[pulsar] = pulsar_dict[pulsar]
#	save_object(final_pulsar_dict, os.path.join(save_dir, "all_pulsar_chain_dict.pkl"))


if __name__ == '__main__':
	main()

