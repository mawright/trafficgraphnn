import logging
from lxml import etree
from collections import OrderedDict, namedtuple
from itertools import chain
import os

import numpy as np
import pandas as pd

import sumolib.net.lane
from trafficgraphnn.utils import E1IterParseWrapper, E2IterParseWrapper, DetInfo
from trafficgraphnn.get_tls_data import get_tls_data

_logger = logging.getLogger(__name__)

_Queueing_Period_Data = namedtuple(
    'QueueingPeriodData',
    ['num_period', 'start_time', 'end_time',
     'list_time_stop', 'list_nVehContrib_stop',
     'list_time', 'list_occupancy', 'list_nVehEntered', 'list_nVehContrib',
     'list_time_e2', 'list_startedHalts_e2', 'list_max_jam_length_m_e2',
     'list_max_jam_length_veh_e2',
     'list_jam_length_sum_e2'])


class SumoNetworkOutputReader(object):
    def __init__(self, sumo_network):
        self.sumo_network = sumo_network
        self.graph = self.sumo_network.get_graph()
        self.net = sumo_network.net
        self.parsed_xml_tls = None

        self.lane_readers = OrderedDict()

        self._hdf_stores = {}

    def add_lane_reader(self, lane_id, reader):
        if lane_id not in self.lane_readers:
            self.lane_readers[lane_id] = reader
        else:
            old_reader = self.lane_readers[lane_id]
            new_out_lane_ids = [out_lane_id for out_lane_id in reader.out_lane_ids
                                if out_lane_id not in old_reader.out_lane_ids]
            for new_id in new_out_lane_ids:
                old_reader.add_out_lane(new_id)

    def get_tls_output_filenames_for_lanes(self):
        return {lane_id: lane.tls_output_filename
                for lane_id, lane in self.lane_readers.items()}

    def reset_phase_timing_info(self):
        for reader in self.lane_readers.values():
            reader.green_intervals = []

    def parse_phase_timings(self):
        tls_output_files = {lane.tls_output_filename for lane
                            in self.lane_readers.values()}

        for xmlfile in tls_output_files:
            parsed = etree.iterparse(xmlfile, tag='tlsSwitch')
            for _, element in parsed:
                try:
                    lane_id = element.attrib['fromLane']
                    start = int(float(element.attrib['begin']))
                    end = int(float(element.attrib['end']))
                except KeyError:
                    _logger.warning(
                        'Could not parse XML element %s. expected a "tlsSwitch"',
                        element)
                    continue
                finally:
                    element.clear()

                try:
                    lane = self.lane_readers[lane_id]
                    lane.add_green_interval(start, end)
                except KeyError: # lane not present
                    pass
        for lane in self.lane_readers.values():
            lane.union_green_intervals()

    def _open_or_get_hdfstore(self, store_filename):
        if store_filename in self._hdf_stores.keys():
            return self._hdf_stores[store_filename]
        else:
            store = pd.HDFStore(store_filename, 'r')
            self._hdf_stores[store_filename] = store
            return store

    def close_hdfstores(self):
        for store in self._hdf_stores.values():
            store.close()
        self._hdf_stores = {}

class SumoLaneOutputReader(object):
    def __init__(self,
                 sumolib_in_lane,
                 out_lane,
                 net_reader=None,
                 raw_hdfstore_filename=None):
        self.sumolib_in_lane = sumolib_in_lane
        self.sumolib_out_lane = out_lane
        self.net_reader = net_reader
        self.out_lane_ids = []

        #dataframes that are loaded once at the beginning
        self.df_e1_adv_detector = pd.DataFrame()
        self.df_e1_stopbar = pd.DataFrame()
        self.df_e2_detector = pd.DataFrame()
        self.df_traffic_lights = pd.DataFrame()
        self.graph = net_reader.graph

        columns_e1_adv = ['time', 'occupancy', 'nVehEntered', 'nVehContrib']
        columns_e1_stopbar = ['time', 'nVehContrib']
        self.curr_e1_adv_detector = pd.DataFrame(columns=columns_e1_adv)
        self.curr_e1_stopbar = pd.DataFrame(columns=columns_e1_stopbar)
        columns_e2 = ['time', 'startedHalts']
        self.curr_e2_detector = pd.DataFrame(columns=columns_e2)

        self.parsed_xml_e1_stopbar_detector = None
        self.parsed_xml_e1_adv_detector = None
        self.parsed_xml_e2_detector = None

        self.lane_id = self.sumolib_in_lane.getID()
        self.networkx_node = self.graph.nodes[self.lane_id]
        self.dash_lane_id = self.lane_id.replace('/', '-')

        self.tls_output_filename = self.get_tls_output_filename()

        self.arr_real_max_queue_length = []
        self.arr_maxJamLengthInMeters = []
        self.arr_maxJamLengthInVehicles = []
        self.arr_maxJamLengthInVehiclesSum = []

        self.arr_phase_start = []
        self.arr_phase_end = []
        self.green_intervals = []
        self.arr_green_phase_start = []
        self.arr_green_phase_end = []

        self.prev_cycle_parsed = None
        self.this_cycle_parsed = None
        self.next_cycle_parsed = None

        self._parse_detector_info()

        if self.net_reader is not None:
            self.net_reader.add_lane_reader(self.lane_id, self)
        self._initalized = False

        if raw_hdfstore_filename is not None:
            self.read_from_hdf = True
            self.raw_hdf_filename = raw_hdfstore_filename
        else:
            self.read_from_hdf = False

    def _parse_detector_info(self):
        det_dict = self.networkx_node['detectors']
        # loop (e1) detectors
        e1_detectors = [DetInfo(k, v) for k, v in det_dict.items()
                        if v['type'] == 'e1Detector']

        # stopbar detector is the one with the largest "pos" (position) value
        e1_by_pos = sorted(e1_detectors, key=lambda x: x.info['pos'])

        stopbar_detector = e1_by_pos[-1]
        adv_detector = e1_by_pos[-2]

        self.stopbar_detector_id = stopbar_detector.id
        self.adv_detector_id = adv_detector.id

        # lane area (e2) detectors
        e2_detectors = [DetInfo(k, v) for k, v in det_dict.items()
                        if v['type'] == 'e2Detector']
        if len(e2_detectors) > 1:
            e2_detector = max(e2_detectors, key=lambda x: x.info['length'])
            _logger.warning(
                'Lane %s seems to have more than one lane-area (e2) detector. '
                'Using %s', self.lane_id, e2_detector)
        else:
            e2_detector = e2_detectors[0]
        self.e2_detector_id = e2_detector.id

        net_dir = os.path.dirname(self.net_reader.sumo_network.netfile)

        self.stopbar_output_file = os.path.join(net_dir,
                                                stopbar_detector.info['file'])
        self.adv_output_file = os.path.join(net_dir,
                                            adv_detector.info['file'])
        self.e2_output_file = os.path.join(net_dir,
                                           e2_detector.info['file'])

    def _initialize_parser(self, start_cycle=0):
        if self.read_from_hdf:
            self._init_hdfstore_parser()
        else:
            self._init_detector_xml_parsers(start_cycle)

    def _init_detector_xml_parsers(self, start_cycle=0):
        self.parsed_xml_e1_stopbar_detector = E1IterParseWrapper(
            self.stopbar_output_file, True, id_subset=self.stopbar_detector_id)
        self.parsed_xml_e1_adv_detector = E1IterParseWrapper(
            self.adv_output_file, True, id_subset=self.adv_detector_id)
        self.parsed_xml_e2_detector = E2IterParseWrapper(
            self.e2_output_file, True, id_subset=self.e2_detector_id)

        if start_cycle > 0:
            prev_start = self.nth_cycle_interval(start_cycle - 1)[0]
        else:
            prev_start = 0
        for _ in self.parsed_xml_e1_stopbar_detector.iterate_until(prev_start):
            continue
        for _ in self.parsed_xml_e1_adv_detector.iterate_until(prev_start):
            continue
        for _ in self.parsed_xml_e2_detector.iterate_until(prev_start):
            continue

        cycle_start = self.nth_cycle_interval(start_cycle)[0]

        self.this_cycle_parsed = self._parse_detector_xmls_until(
            cycle_start, start_cycle - 1, prev_start)
        self.next_cycle_parsed = self._parse_xmls_for_cycle(start_cycle)

        self._initalized = True

    def _init_hdfstore_parser(self, start_cycle=0):
        self.store = self.net_reader._open_or_get_hdfstore(self.raw_hdf_filename)

        if start_cycle > 0:
            self.this_cycle_parsed = self._read_cycle_from_hdf(start_cycle - 1)
        else:
            first_queue_period_start = self.nth_cycle_interval(0)[0]
            self.this_cycle_parsed = self._read_period_from_hdf(-1, 0, first_queue_period_start)

        self.next_cycle_parsed = self._read_cycle_from_hdf(start_cycle)
        self._initalized = True

    def _read_cycle_from_hdf(self, num_cycle):
        start_time, end_time = self.nth_cycle_interval(num_cycle)
        return self._read_period_from_hdf(num_cycle, start_time, end_time)

    def _read_period_from_hdf(self, num_cycle, start_time=None, end_time=None):
        if start_time is None or end_time is None:
            start_time, end_time = self.nth_cycle_interval(num_cycle)

        cycle_indexer = slice(start_time, end_time-1)

        # advance detector
        adv_detector_string = f'raw_xml/{self.adv_detector_id}'
        list_time = list(self.store[adv_detector_string].loc[cycle_indexer].index)
        list_occupancy = list(self.store[adv_detector_string].loc[cycle_indexer]['occupancy'])
        list_nVehEntered = list(self.store[adv_detector_string].loc[cycle_indexer]['nVehEntered'])
        list_nVehContrib = list(self.store[adv_detector_string].loc[cycle_indexer]['nVehContrib'])

        # stopbar detector
        stopbar_string = f'raw_xml/{self.stopbar_detector_id}'
        list_time_stop = list(self.store[stopbar_string].loc[cycle_indexer].index)
        list_nVehContrib_stop = list(self.store[stopbar_string].loc[cycle_indexer]['nVehContrib'])

        # e2 detector
        e2_string = f'raw_xml/{self.e2_detector_id}'
        list_time_e2 = list(self.store[e2_string].loc[cycle_indexer].index)
        list_startedHalts_e2 = list(self.store[e2_string].loc[cycle_indexer]['startedHalts'])
        list_max_jam_length_m_e2 = list(self.store[e2_string].loc[cycle_indexer]['maxJamLengthInMeters'])
        list_max_jam_length_veh_e2 = list(self.store[e2_string].loc[cycle_indexer]['maxJamLengthInVehicles'])
        list_jam_length_sum_e2 = list(self.store[e2_string].loc[cycle_indexer]['jamLengthInMetersSum'])

        interval_data = _Queueing_Period_Data(
            num_cycle, start_time=start_time, end_time=end_time,
            list_time_stop=list_time_stop, list_nVehContrib_stop=list_nVehContrib_stop,
            list_time=list_time, list_occupancy=list_occupancy,
            list_nVehEntered=list_nVehEntered, list_nVehContrib=list_nVehContrib,
            list_time_e2=list_time_e2, list_startedHalts_e2=list_startedHalts_e2,
            list_max_jam_length_m_e2=list_max_jam_length_m_e2,
            list_max_jam_length_veh_e2=list_max_jam_length_veh_e2,
            list_jam_length_sum_e2=list_jam_length_sum_e2)

        return interval_data

    def _parse_xmls_for_cycle(self, num_cycle):
        if self.this_cycle_parsed is not None and num_cycle < self.this_cycle_parsed.num_period:
            self._init_detector_xml_parsers(num_cycle)
            return self.this_cycle_parsed
        start, end = self.nth_cycle_interval(num_cycle)

        ### assertions only valid under fixed cycle timing
        try:
            _, next_end = self.nth_cycle_interval(num_cycle + 1)

            next_green = next_end # old naming

            if num_cycle == 0:
                prev_start = 0
            else:
                prev_start, _ = self.nth_cycle_interval(num_cycle - 1)

            prev_red = prev_start # old naming
            start_old = int(self.phase_start + num_cycle*self.phase_length) #seconds #start begins with red phase
            end_old = start_old + self.phase_length #seconds #end is end of green phase
            assert start_old == start
            assert end_old == end

            assert prev_red == start - self.phase_length or start - self.phase_length <= 0
            assert next_green == end + self.phase_length
        except AttributeError: # self.phase_start and similar items not initialized
            pass
        except IndexError: # nth_cycle_interval(num_cycle + 1) breaks, we are at the end
            pass
        #####

        # see if the iterparse xmls are seeked to the right time
        if (not self.parsed_xml_e1_adv_detector.interval_begin() == start
            or not self.parsed_xml_e1_stopbar_detector.interval_begin() == start
            or not self.parsed_xml_e2_detector.interval_begin() == start
        ):
            self._init_detector_xml_parsers(start)
            assert (self.parsed_xml_e1_adv_detector.interval_begin() == start
                    and self.parsed_xml_e1_stopbar_detector.interval_begin() == start
                    and self.parsed_xml_e2_detector.interval_begin() == start)

        interval_data = self._parse_detector_xmls_until(end, num_cycle, start)

        return interval_data

    def _cycle_cycles(self):
        self.prev_cycle_parsed = self.this_cycle_parsed
        self.this_cycle_parsed = self.next_cycle_parsed
        assert self.this_cycle_parsed is not None
        last_cycle = self.this_cycle_parsed.num_period
        if self.read_from_hdf:
            self.next_cycle_parsed = self._read_cycle_from_hdf(last_cycle + 1)
        else:
            self.next_cycle_parsed = self._parse_xmls_for_cycle(last_cycle + 1)

    def add_out_lane(self, out_lane_id):
        if isinstance(out_lane_id, sumolib.net.Lane):
            out_lane_id = out_lane_id.getID()
        self.out_lane_ids.append(out_lane_id)

    def add_green_interval(self, start_time, end_time):
        self.green_intervals.append((start_time, end_time))

    def union_green_intervals(self, assign_fixed_timing_heuristic=True):
        intervals = []
        for begin, end in sorted(self.green_intervals):
            if intervals and intervals[-1][1] >= begin - 1:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([begin, end])
        self.green_intervals = intervals

        if assign_fixed_timing_heuristic:
            self._estimate_fixed_cycle_timings()

    def parse_cycle_data(self, num_phase):
        #parse the data from the dataframes and write to the arrays in every cycle
        if not self._initalized:
            self._initialize_parser(num_phase)

        start, end = self.nth_cycle_interval(num_phase)

        self._cycle_cycles()
        assert num_phase == self.this_cycle_parsed.num_period
        assert start == self.this_cycle_parsed.start_time

        #using the last THREE phases, because they are needed to estimate breakpoints from the second last one!
        #one in future for Breakpoint C, one in past for simple input-output method

        three_periods = [self.prev_cycle_parsed, self.this_cycle_parsed, self.next_cycle_parsed]

        list_time = list(chain(*[per.list_time for per in three_periods]))
        list_occupancy = list(chain(*[per.list_occupancy for per in three_periods]))
        list_nVehEntered = list(chain(*[per.list_nVehEntered for per in three_periods]))
        list_nVehContrib = list(chain(*[per.list_nVehContrib for per in three_periods]))

        self.curr_e1_adv_detector = pd.DataFrame(
                {'time': list_time, 'occupancy': list_occupancy,
                 'nVehEntered': list_nVehEntered, 'nVehContrib': list_nVehContrib})

        list_time_stop = list(chain(*[per.list_time_stop for per in three_periods]))
        list_nVehContrib_stop = list(chain(*[per.list_nVehContrib_stop for per in three_periods]))

        self.curr_e1_stopbar = pd.DataFrame(
            {'time': list_time_stop, 'nVehContrib': list_nVehContrib_stop})

        self.curr_e1_adv_detector.set_index('time', inplace=True)
        self.curr_e1_stopbar.set_index('time', inplace=True)

        ### parse e2 detector data from xml file ###
        list_time_e2 = list(chain(*[per.list_time_e2 for per in three_periods]))
        list_startedHalts_e2 = list(chain(*[per.list_startedHalts_e2 for per in three_periods]))
        list_max_jam_length_m_e2 = list(chain(*[per.list_max_jam_length_m_e2 for per in three_periods]))
        list_max_jam_length_veh_e2 = list(chain(*[per.list_max_jam_length_veh_e2 for per in three_periods]))
        list_jam_length_sum_e2 = list(chain(*[per.list_jam_length_sum_e2 for per in three_periods]))

        self.curr_e2_detector = pd.DataFrame(
                {'time': list_time_e2,
                 'startedHalts': list_startedHalts_e2,
                 'maxJamLengthInMeters': list_max_jam_length_m_e2,
                 'maxJamLengthInVehicles': list_max_jam_length_veh_e2,
                 'jamLengthInMetersSum': list_jam_length_sum_e2})
        self.curr_e2_detector.set_index('time', inplace=True)

        return start, end, self.curr_e1_stopbar, self.curr_e1_adv_detector, self.curr_e2_detector

    def _parse_detector_xmls_until(self, end_time, num_cycle, start_time):

        # advance detector
        list_time = []
        list_occupancy = []
        list_nVehEntered = []
        list_nVehContrib = []
        for interval in self.parsed_xml_e1_adv_detector.iterate_until(end_time):
            interval_begin = float(interval.attrib.get('begin'))
            det_id = interval.attrib.get('id')
            if det_id == self.adv_detector_id:
                list_time.append(interval_begin)
                list_occupancy.append(float(interval.attrib.get('occupancy')))
                list_nVehEntered.append(float(interval.attrib.get('nVehEntered')))
                list_nVehContrib.append(float(interval.attrib.get('nVehContrib')))

        # stopbar detector
        list_time_stop = []
        list_nVehContrib_stop = []
        for interval in self.parsed_xml_e1_stopbar_detector.iterate_until(end_time):
            interval_begin = float(interval.attrib.get('begin'))
            det_id = interval.attrib.get('id')
            if det_id == self.stopbar_detector_id:
                list_time_stop.append(interval_begin)
                list_nVehContrib_stop.append(float(interval.attrib.get('nVehContrib')))

        # e2 detector
        list_time_e2 = []
        list_startedHalts_e2 = []
        list_max_jam_length_m_e2 = []
        list_max_jam_length_veh_e2 = []
        list_jam_length_sum_e2 = []

        for interval in self.parsed_xml_e2_detector.iterate_until(end_time):
            interval_begin = float(interval.attrib.get('begin'))
            det_id = interval.attrib.get('id')
            if det_id == self.e2_detector_id:
                list_time_e2.append(interval_begin)
                list_startedHalts_e2.append(float(interval.attrib.get('startedHalts')))
                list_max_jam_length_m_e2.append(float(interval.attrib.get('maxJamLengthInMeters')))
                list_max_jam_length_veh_e2.append(float(interval.attrib.get('maxJamLengthInVehicles')))
                list_jam_length_sum_e2.append(float(interval.attrib.get('jamLengthInMetersSum')))

        interval_data = _Queueing_Period_Data(
            num_cycle, start_time=start_time, end_time=end_time,
            list_time_stop=list_time_stop, list_nVehContrib_stop=list_nVehContrib_stop,
            list_time=list_time, list_occupancy=list_occupancy,
            list_nVehEntered=list_nVehEntered, list_nVehContrib=list_nVehContrib,
            list_time_e2=list_time_e2, list_startedHalts_e2=list_startedHalts_e2,
            list_max_jam_length_m_e2=list_max_jam_length_m_e2,
            list_max_jam_length_veh_e2=list_max_jam_length_veh_e2,
            list_jam_length_sum_e2=list_jam_length_sum_e2)

        return interval_data

    def nth_cycle_interval(self, n):
        """Returns the nth queueing and discharging time interval for this lane: (start, end)

        "start" corresponds to start of red phase (end of previous green phase).
        "end" corresponds to end of green phase

        :param n: Index of desired cycle time interval
        :type n: int
        :return: Start and end of cycle time interval
        :rtype: Tuple of ints
        """

        assert n < len(self.green_intervals)
        assert n >= 0
        # if n == 0:
        #     start = 0
        # else:
        #     start = self.green_intervals[n - 1][1]
        # end = self.green_intervals[n][0]
        # return start, end
        return self.green_intervals[n][1], self.green_intervals[n + 1][1]

    def _estimate_fixed_cycle_timings(self):
        first_green, second_green = self.green_intervals[0], self.green_intervals[1]
        self.duration_green_light = first_green[1] - first_green[0]
        self.phase_length = second_green[0] - first_green[0]
        self.phase_start = first_green[1]

        # estimating tls data!

        if self.net_reader.parsed_xml_tls == None:
            self.net_reader.parsed_xml_tls = etree.parse(self.tls_output_filename) # TODO move to own function that constructor calls

        (phase_start_old, phase_length_old, duration_green_light_old
                    ) = get_tls_data(self.net_reader.parsed_xml_tls, self.lane_id)

        if phase_start_old != self.phase_start:
            _logger.warning('Lane %s: phase_start_old = %g, phase_start = %g',
                            self.lane_id, phase_start_old, self.phase_start)
        if phase_length_old != self.phase_length:
            _logger.warning('Lane %s: phase_length_old = %g, phase_length = %g',
                            self.lane_id, phase_length_old, self.phase_length)
        if duration_green_light_old != self.duration_green_light:
            _logger.warning('Lane %s: duration_green_light_old = %g, duration_green_light = %g',
                            self.lane_id, duration_green_light_old, self.duration_green_light)

    def nth_green_phase_intervals(self, n):
        """
        Returns the nth time interval of green light for this lane: (start, end)

        "start" corresponds to start of green phase
        "end" corresponds to end of green phase

        :param n: Index of desired cycle time interval
        :type n: int
        :return: Start and end of cycle time interval
        :rtype: Tuple of ints
        """
        assert n <= len(self.green_intervals)
        assert n >= 0

        return self.green_intervals[n+1][0], self.green_intervals[n+1][1]

    def get_tls_output_filename(self):
        """Return the filename of the tls-switch output file from the nx graph.

        :return: tls-switch filename
        :rtype: string
        """

        return os.path.join(os.path.dirname(self.net_reader.sumo_network.netfile),
                            self.graph.edges[self.lane_id,
                                             self.sumolib_out_lane.getID()
                                            ]['tls_output_info']['dest'])

    def get_phase_length(self):
        return self.phase_length

    def get_num_phases(self):
        """Return the number of phases (number of green intervals) from tls data

        :return: number of phases
        :rtype: int
        """
        return len(self.green_intervals)

    def phases_before(self, end_time=np.inf):
        """Return the phases (green intervals before a certain time)

        :param end_time: Upper limit of green phase starts, defaults to np.inf
        :param end_time: numeric, optional
        """
        if end_time == np.inf:
            return self.green_intervals
        return [interval for interval in self.green_intervals
                if interval[0] < end_time]

    def get_num_phases_before(self, end_time=np.inf):
        """Return the number of phases (number of green intervals) before a certain time

        :param end_time: Time to stop counting phases, defaults to np.inf
        :param end_time: numeric, optional
        """
        if end_time == np.inf:
            return self.get_num_phases()
        return len(self.phases_before(end_time))

    def get_lane_ID(self):
        return self.sumolib_in_lane.getID()
