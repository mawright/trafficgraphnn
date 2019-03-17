import argparse
import os
import numpy as np

from trafficgraphnn.sumo_network import SumoNetwork
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.preprocessing.preprocess import run_preprocessing

def main(
    net_name,
    num_simulations,
    end_time=3600,
    period=.4,
    seed=1234,
    grid_number=3,
    grid_length=750,
    num_lanes=3,
):
    np.random.seed(seed)
    net_dir = os.path.join('data/networks', net_name)
    net_filename = os.path.join(net_dir, f'{net_name}.net.xml')

    if os.path.exists(net_filename):
        sn = SumoNetwork.from_preexisting_directory(net_dir)
    else:
        config = ConfigGenerator(net_name=net_name)

        config.gen_grid_network(
            grid_number=grid_number, grid_length=grid_length,
            num_lanes=num_lanes)
        config.gen_rand_trips(
            period=period, binomial=None, end_time=end_time)

        config.gen_e1_detectors(distance_to_tls=[5, 125], frequency=1)
        config.gen_e2_detectors(distance_to_tls=0, frequency=1)
        config.define_tls_output_file()
        sn = SumoNetwork.from_gen_config(config)

    for _ in range(num_simulations):
            sn.config_gen.gen_rand_trips(
                period=period, binomial=None, end_time=end_time)
            sn.run()
            run_preprocessing(sn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net_name', type=str, help='Name for Sumo network.')
    parser.add_argument('num_simulations', type=int,
                        help='Number of simulations to run.')
    parser.add_argument('--end_time', '-e', type=int, default=3600,
                        help='End time to stop generating trips.')
    parser.add_argument('--period', '-p', type=float, default=.4,
                        help='Period for random trips.')
    parser.add_argument('--grid_number', '-n', type=int, default=3,
                        help='Size of grid network.')
    parser.add_argument('--grid_length', '-gl', type=int, default=750,
                        help='Length of grid roads in m.')
    parser.add_argument('--num_lanes', '-nl', type=int, default=3,
                        help='Number of lanes per road.')
    parser.add_argument('--seed', '-s', type=int, default=1234,
                        help="Random seed.")

    args = parser.parse_args()
    main(args.net_name, args.num_simulations,
         end_time=args.end_time,
         period=args.period,
         grid_number=args.grid_number,
         grid_length=args.grid_length,
         num_lanes=args.num_lanes,
         seed=args.seed,
    )
