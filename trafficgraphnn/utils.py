from __future__ import absolute_import, print_function, division

import os
import sys
import collections

from lxml import etree


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


def xml_to_list_of_dicts(
    xml_file, tags_to_filter=None, attributes_to_get=None
):
    if attributes_to_get is None:
        get_all = True
    else:
        get_all = False

    data = etree.parse(xml_file)
    all_records = []
    for child in data.iter(tags_to_filter):
        if get_all:
            record = dict(child.items())
        else:
            record = {}
            for attr in attributes_to_get:
                if attr in child.keys():
                    record[attr] = child.get(attr)
        all_records.append(record)

    return all_records


def iterfy(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, basestring):
        return x
    else:
        return (x,)
