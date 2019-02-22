import os

import lxml
import numpy as np
import pandas as pd

from trafficgraphnn import SumoNetwork
from trafficgraphnn.utils import sumo_output_xmls_to_hdf
from trafficgraphnn.preprocess_data import PreprocessData

def load_sumo_output_data(sumo_network):
    output_dir = sumo_network.detector_data_path
    output_hdf = sumo_output_xmls_to_hdf(output_dir)

    green_df = PreprocessData(sumo_network).read_light_timings()
