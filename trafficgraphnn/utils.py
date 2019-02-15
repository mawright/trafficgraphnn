import collections
import logging
import os
import re
import sys
from itertools import zip_longest, chain, repeat, tee
import warnings
import tables

import numpy as np
import pandas as pd
import six
from lxml import etree
from tables.exceptions import NoSuchNodeError

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


DetInfo = collections.namedtuple('det_info', ['id', 'info'])


class IterParseWrapper(object):
    _tag = None
    _schema_file = None
    def __init__(self, xml_file, validate=False, id_subset=None):
        if validate:
            try:
                schema_file = self._schema_file
                schema = etree.XMLSchema(file=schema_file)
                tree = etree.iterparse(xml_file,
                                       schema=schema,
                                       events=('start', 'end'))
            except etree.XMLSchemaParseError:
                _logger.warning(
                    'Error in xml validation of %s, skipping validation.',
                    xml_file,
                    exc_info=True)
                tree = etree.iterparse(xml_file, events=('start', 'end'))
        else:
            tree = etree.iterparse(xml_file, events=('start', 'end'))
        self.tree = tree
        _, self.root = six.next(tree)
        if id_subset is not None:
            self.get_next = self.__get_next_in_subset
            self.id_subset = iterfy(id_subset)
        else:
            self.get_next = self.__get_next_unfiltered
        self.get_next()

    def __get_next_unfiltered(self):
        while True:
            event, elem = six.next(self.tree)
            if event == 'end' and elem.tag == self._tag:
                self.item = elem
                return

    def __get_next_in_subset(self):
        while True:
            event, elem = six.next(self.tree)
            if (event == 'end'
                and elem.tag == self._tag
                and elem.get('id') in self.id_subset
            ):
                self.item = elem
                return
            else:
                # elem.clear()
                self.root.clear()

    def iterate_until(self, stop_time):
        while self.interval_end() <= stop_time:
            yield self.item
            # self.item.clear()
            self.root.clear()
            try:
                self.get_next()
            except StopIteration:
                return

    def interval_end(self):
        return float(self.item.attrib.get('end'))

    def interval_begin(self):
        return float(self.item.attrib.get('begin'))


class TLSSwitchIterParseWrapper(IterParseWrapper):
    _tag = 'tlsSwitch'
    _schema_file = os.path.join(get_sumo_dir(), 'data', 'xsd', 'tlsswitches_file.xsd')

class E1IterParseWrapper(IterParseWrapper):
    _tag = 'interval'
    _schema_file = os.path.join(get_sumo_dir(), 'data', 'xsd', 'det_e1_file.xsd')


class E2IterParseWrapper(IterParseWrapper):
    _tag = 'interval'
    _schema_file = os.path.join(get_sumo_dir(), 'data', 'xsd', 'det_e2_file.xsd')


_col_dtype_key = {
    'begin': float,
    'end': float,
    'id': str,
    'nVehContrib': int,
    'flow': float,
    'occupancy': float,
    'speed': float,
    'length': float,
    # e2
    'sampledSeconds': float,
    'nVehEntered': int,
    'nVehLeft': int,
    'nVehSeen': int,
    'meanSpeed': float,
    'meanTimeLoss': float,
    'meanOccupancy': float,
    'maxOccupancy': float,
    'meanMaxJamLengthInVehicles': float,
    'meanMaxJamLengthInMeters': float,
    'maxJamLengthInVehicles': int,
    'maxJamLengthInMeters': float,
    'jamLengthInVehiclesSum': int,
    'jamLengthInMetersSum': float,
    'meanHaltingDuration': float,
    'maxHaltingDuration': float,
    'haltingDurationSum': float,
    'meanIntervalHaltingDuration': float,
    'maxIntervalHaltingDuration': float,
    'intervalHaltingDurationSum': float,
    'startedHalts': float,
    'meanVehicleNumber': float,
    'maxVehicleNumber': int,
    # tlsSwitch
    'programID': str,
    'duration': float,
    'fromLane': str,
    'toLane': str
}


def _append_to_store(store, buffer, all_ids):
    converter = {col: _col_dtype_key[col]
                      for col in buffer.keys()
                      if col in _col_dtype_key}
    df = pd.DataFrame.from_dict(buffer)
    df = df.astype(converter)
    df = df.set_index('begin')
    for i in all_ids:
        # sub_df = df.loc[df['id'] == i]
        # sub_df = sub_df.set_index('begin')
        sub_df = df.query(f"id == '{i}'")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', tables.NaturalNameWarning)
            store.append(f'raw_xml/{i}', sub_df)
        assert len(store[f'raw_xml/{i}'].loc[0].shape) == 1


def xml_to_df_hdf(parser,
                  store_filename,
                  complevel=7,
                  complib='zlib',
                  start_time=0,
                  end_time=np.inf,
                  buffer_size=1e5):
    buffer = collections.defaultdict(list)
    i = 0
    all_ids = set()
    with pd.HDFStore(
        store_filename, complevel=complevel, complib=complib) as store:
        for _ in parser.iterate_until(start_time):
            pass
        for row in parser.iterate_until(end_time):
            for k, v in row.items():
                buffer[k].append(v)
            all_ids.add(row.get('id'))
            i += 1
            if i >= buffer_size:
                _append_to_store(store, buffer, all_ids)
                buffer = collections.defaultdict(list)
                i = 0
        _append_to_store(store, buffer, all_ids)


def get_preprocessed_filenames(directory):
    try:
        return [os.path.join(directory, f)
                for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))
                and re.match(r'sim_\d+.h5', os.path.basename(f))]
    except FileNotFoundError:
        return []


def get_sim_numbers_in_preprocess_store(store, lane_list=None):
    if lane_list is None:
        lane_list = store['A_downstream'].columns

    def sim_numbers_for_lane(lane):
            X_subelements = dir(store.root.__getattr__(lane).X)
            samples = filter(lambda e: re.match(r'_\d{4,5}', e), X_subelements)
            return list(map(lambda e: int(re.search(r'\d{4,5}', e).group()), samples))

    def sim_numbers_for_lane_2(lane):
        query_string = '{}/X/_{:04}'
        i = 1
        try:
            while True:
                X = store.get(query_string.format(lane, i))
                i += 1
        except KeyError:
            pass
        return list(range(1, i))

    try:
        sim_numbers = sim_numbers_for_lane_2(list(lane_list)[0])
    except NoSuchNodeError:
        return []

    # assert all((sim_numbers_for_lane_2(lane) == sim_numbers for lane in lane_list))
    return sim_numbers


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


def in_interval(value, interval):
    assert(len(interval)) == 2
    return interval[0] <= value <= interval[1]


def iterfy(x):
    if isinstance(x, collections.Iterable) and type(x) not in six.string_types:
        return x
    else:
        return (x,)


def string_list_decode(x):
    x = iterfy(x)
    return [s.decode() if isinstance(s, six.binary_type) else s for s in x]


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    # From itertools recipe page
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def paditerable(iterable, pad_value=None):
    """Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.
    """
    # From itertools recipe page
    return chain(iterable, repeat(pad_value))


def flatten(listOfLists):
    "Flatten one level of nesting"
    # From itertools recipe page
    return chain.from_iterable(listOfLists)


def pairwise_exhaustive(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    # From itertools recipe page (modified)
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)


def broadcast_lists(iterables):
    len0 = len(iterables[0])
    if not all(len(x) in [1, len0] for x in iterables):
        raise ValueError('Inputs should all be the same length or length 1')

    for i, iterable in enumerate(iterables):
        if len(iterable) == 1:
            iterables[i] = list(iterables[i]) * len0

    return iterables
