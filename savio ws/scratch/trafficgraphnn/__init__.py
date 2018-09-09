from __future__ import absolute_import, print_function, division
import logging
from logging.config import fileConfig
import os

from trafficgraphnn.utils import append_sumo_tools_dir

configfile = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'logging.conf')
fileConfig(configfile)
logger = logging.getLogger(__name__)

append_sumo_tools_dir()
