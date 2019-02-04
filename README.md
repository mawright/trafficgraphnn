# Graph-structured neural networks for road network data
This repository contains research code for neural networks for graph-structured data.

This code is still being actively developed and will be updated. The most up-to-date version can always be found at https://github.com/mawright/trafficgraphnn.

The most useful items for general graph data problems are the Keras layers (see below).

## Requirements
The traffic simulator SUMO (http://sumo.dlr.de/) is required. For installation info, see http://sumo.dlr.de/wiki/Installing.

SUMO comes with a large number of Python scripts in its "tools" directory.
This code depends on many of them.
After installing SUMO, set the environmental variable `SUMO_HOME` to your SUMO installation directory (e.g., `usr/local/sumo/`) so that the Python interpreter can find them.

## Learning on SUMO networks
The library provides functions to generate road networks and random traffic simulations for learning. The script `demo_gen_data.py` shows an example of how to generate the SUMO road network, generate simulation configurations, and run the simulations to generate data.

The file `demo_train_script.py` shows a very rough example of building a Keras model for the road network and learning on road network data. 

## Keras layers
Of potential interest are the Keras layers we have developed. They are located in the module `trafficgraphnn.layers`. Some completed and tested layers are:
- `BatchGraphAttention`: Computes featurizations of graph elements based on their own and
their neighbors' features using neural attention.
Takes two inputs: 
    - Data `X` of shape `(batch, nodes, features)`
    - Adjacency matrices `A` of shape `(batch, nodes, nodes)`
- `TimeDistributedMultiInput`: Generalization of Keras's `TimeDistributed` layer to allow multiple inputs (e.g., for time-distributed `BatchGraphAttention` layers).
- `DenseCausalAttention`: RNN decoder attention wrapper based off a proposed Keras Attention API. To be used in timeseries tasks where the prediction at time _t_  should be a function only of data from timesteps prior to _t_.

More layers are still in development.