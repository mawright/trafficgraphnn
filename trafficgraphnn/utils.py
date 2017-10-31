from __future__ import absolute_import, print_function, division

import os
import sys


def get_sumo_dir():
    return os.path.join(os.environ['SUMO_HOME'])


def get_sumo_tools_dir():
    tools = os.path.join(get_sumo_dir(), 'tools')
    return tools


def append_sumo_tools_dir():
    if 'SUMO_HOME' in os.environ:
        tools_dir = get_sumo_tools_dir()
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")


def get_edge_neighbors(edge):
    return edge.getIncoming(), edge.getOutgoing()
