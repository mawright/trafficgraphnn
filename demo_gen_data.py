import argparse
import os

from trafficgraphnn.sumo_network import SumoNetwork
from trafficgraphnn.genconfig import ConfigGenerator
from trafficgraphnn.preprocess_data import PreprocessData

def main(n):

    net_name = f'testnet3x3_{n}'

    net_dir = os.path.join('data/networks', net_name)
    net_filename = os.path.join(net_dir, f'{net_name}.net.xml')

    end_time = 3600
    period = 0.4

    if os.path.exists(net_filename):
        sn = SumoNetwork(net_filename,
                         routefile=os.path.join(net_dir, f'{net_name}_random.routes.xml'),
                         addlfiles=[os.path.join(net_dir, f'{net_name}_e1.add.xml'),
                                    os.path.join(net_dir, f'{net_name}_e2.add.xml'),
                                    os.path.join(net_dir, 'tls_output.add.xml')])
    else:
        config = ConfigGenerator(net_name=net_name)

        config.gen_grid_network(
            grid_number=3, grid_length=600, simplify_tls = False)
        config.gen_rand_trips(
            period=period, binomial=None, end_time=end_time)

        config.gen_e1_detectors(distance_to_tls=[5, 125], frequency=1)
        config.gen_e2_detectors(distance_to_tls=0, frequency=1)
        config.define_tls_output_file()
        sn = SumoNetwork.from_gen_config(config)

    num_files = 8
    num_sims_per_file = 8

    for f in range(num_files):
        preproc = PreprocessData(sn)
        for sim in range(num_sims_per_file):
            print(f'Running sim {sim} for file {f}')
            sn.config_gen.gen_rand_trips(period=period, binomial=None, end_time=end_time)
            sn.run()
            preproc.read_data()
            preproc.run_liu_method()
            preproc.extract_liu_results()
            preproc.write_per_lane_table(delete_intermediate_tables=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net_number', type=int)

    args = parser.parse_args()
    main(args.net_number)
