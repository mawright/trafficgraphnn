from __future__ import absolute_import, print_function, division

import os
import sys
import collections
import logging

import six
import pandas as pd
from lxml import etree


_logger = logging.getLogger(__name__)


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


def get_net_name(netfile):
    return os.path.basename(
        os.path.splitext(os.path.splitext(netfile)[0])[0])


def get_net_dir(netfile):
    return os.path.dirname(os.path.realpath(netfile))


def load_data(network_name, output_tag):
    pass


class IterParseWrapper(object):
    _tag = None
    _schema_file = None
    def __init__(self, xml_file, validate=False):
        if validate:
            try:
                schema_file = self._schema_file
                schema = etree.XMLSchema(file=schema_file)
                tree = etree.iterparse(xml_file, schema=schema, tag=self._tag)
            except etree.XMLSchemaParseError:
                _logger.warning(
                    'Error in xml validation of %s, skipping validation.',
                    xml_file,
                    exc_info=True)
                tree = etree.iterparse(xml_file, tag=self._tag)
        else:
            tree = etree.iterparse(xml_file, tag=self._tag)
        self.tree = tree
        self.get_next()

    def get_next(self):
        _, self.item = six.next(self.tree)

    def iterate_until(self, stop_time):
        while self.interval_end() <= stop_time:
            yield self.item
            self.item.clear()
            self.get_next()

    def interval_end(self):
        return float(self.item.attrib.get('end'))

    def interval_begin(self):
        return float(self.item.attrib.get('begin'))


class E1IterParseWrapper(IterParseWrapper):
    _tag = 'interval'
    _schema_file = os.path.join(get_sumo_dir(), 'data', 'xsd', 'det_e1_file.xsd')


class E2IterParseWrapper(IterParseWrapper):
    _tag = 'interval'
    _schema_file = os.path.join(get_sumo_dir(), 'data', 'xsd', 'det_e2_file.xsd')


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


def parse_detector_output_xml(data_file, ids=None, fields=None):
    parsed = etree.iterparse(data_file, tag='interval')

    records = {}

    for _, element in parsed:
        det_id = element.attrib['id']
        if ids is None or det_id in ids:
            if fields is None:
                record = {col: element.attrib[col]
                          for col in element.keys()
                          if col not in ['begin', 'id']}
            else:
                record = {col: element.attrib[col]
                          for col in fields
                          if col in element.keys()}

            records[(int(round(float(element.attrib['begin']))), det_id,
                     )] = record

    df = pd.DataFrame.from_dict(records, orient='index', dtype=float)
    df.index.set_names(['time', 'det_id'], inplace=True)

    return df


def parse_tls_output_xml(data_file):
    parsed = etree.iterparse(data_file, tag='tlsSwitch')

    records = []

    for _, element in parsed:
        records.append(
            (element.attrib['id'],
             element.attrib['fromLane'],
             element.attrib['toLane'],
             element.attrib['programID'],
             float(element.attrib['begin']),
             float(element.attrib['end']),
             float(element.attrib['duration']))
        )

    df = pd.DataFrame.from_records(
        records,
        columns=[
            'tls_id', 'fromLane', 'toLane', 'programID',
            'begin', 'end', 'duration'])
    df.set_index(['tls_id', 'fromLane', 'toLane'], inplace=True)

    return df


def verify_xml_schema(xml_file):
    tools_dir = get_sumo_tools_dir()
    schemacheck_py_file = os.path.join(
        tools_dir, 'xml', 'schemaCheck.py')

    # schemaCheck.py errors when you send it a valid tls output file?
    raise NotImplementedError


def in_interval(value, interval):
    assert(len(interval)) == 2
    return interval[0] <= value <= interval[1]


def iterfy(x):
    if isinstance(x, collections.Iterable) and type(x) not in six.string_types:
        return x
    else:
        return (x,)
