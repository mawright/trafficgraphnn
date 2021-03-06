import logging
import os
from collections import OrderedDict, namedtuple

import numpy as np
import pandas as pd
import math

from trafficgraphnn.utils import iterfy
from trafficgraphnn.sumo_output_reader import SumoNetworkOutputReader, SumoLaneOutputReader


_logger = logging.getLogger(__name__)

JAM_DENSITY = 0.13333  # hardcoded jam density value (veh/meter)

class LiuEtAlRunner(object):
    def __init__(self, sumo_network,
                 input_data_hdf_file=None,
                 lane_subset=None,
                 time_window=None,
                 store_while_running = True,
                 use_started_halts = False,
                 sim_num = 0,
                 test_data = False):
        self.reader = SumoNetworkOutputReader(sumo_network)
        self.sumo_network = sumo_network
        self.df_estimation_results = pd.DataFrame()
        self.store_while_running = store_while_running
        self.use_started_halts = use_started_halts
        self.sim_num = sim_num
        self.test_data = test_data
        self.input_data_hdf_file = input_data_hdf_file
        # verify all lanes requested are actually in the network
        if lane_subset is None:
            # process all lanes
            self.lane_subset = self.sumo_network.graph.nodes
        else:
            lane_subset = iterfy(lane_subset)
            lanes_not_in_graph = [
                lane for lane in lane_subset
                if lane not in self.sumo_network.graph.nodes
            ]
            if len(lanes_not_in_graph) > 0:
                _logger.warning(
                    'The following lanes are not present in the supplied'
                    ' network and will be ignored: {}'.format(
                        lanes_not_in_graph)
                )
                self.lane_subset = [
                    lane for lane in lane_subset
                    if lane in self.sumo_network.graph.nodes
                ]

        # create LiuIntersection objects for each intersection
        self.liu_intersections = [
            LiuIntersection(tls, self, time_window=time_window)
            for tls in self.net.getTrafficLights()
        ]
        _logger.info('Created %s LiuIntersection objects',
                     len(self.liu_intersections))

        list_of_lane_dicts = [i.liu_lanes for i in self.liu_intersections]
        self.liu_lanes = {lane_id: lane for lane_dict in list_of_lane_dicts
                          for lane_id, lane in lane_dict.items()}

        self.parse_phase_timings()
        self._init_max_per_green_estimates()
        self._init_detector_parsers()

    @property
    def graph(self):
        return self.reader.graph

    @property
    def net(self):
        return self.reader.net

    @property
    def parsed_xml_tls(self):
        return self.reader.parsed_xml_tls

    def _init_max_per_green_estimates(self):
        for lane in self.liu_lanes.values():
            lane._init_max_per_green_estimate()

    def parse_phase_timings(self):
        self.reader.parse_phase_timings()

    def _init_detector_parsers(self):
        for lane in self.liu_lanes.values():
            lane._init_detector_parsers()

    def run_up_to_phase(self, max_num_phase):
        # iterate on the single-step methods for each intersection until
        # reaching the given time
        # for num_phase in range(1, max_num_phase):
        for num_phase in range(0, max_num_phase):
            _logger.info(
                'Running estimation for every lane in every intersection in phase %g',
                num_phase)
            self.df_estimation_results = pd.DataFrame() #reset df for case of online storage
            for intersection in self.liu_intersections:
                intersection.run_next_phase(num_phase)
                if self.store_while_running == True:
                    intersection.add_estimation_to_df(self.store_while_running, num_phase)
            if self.store_while_running == True:
                self.append_results(num_phase)

        if self.store_while_running == False:
            self.store_results(unload_data = True)

    def run_next_phase(self, num_phase):
        # run the single-phase method for each intersection
        for intersection in self.liu_intersections:
            intersection.run_next_phase(num_phase)

    def plot_results_every_lane(self, show_plot=True, show_infos=True):
        for intersection in self.liu_intersections:
            intersection.plot_results(show_plot, show_infos)

    def get_total_MAPE_per_net(self):
        sum_MAPE_IO = 0
        sum_MAPE_liu = 0
        cnt = 0
        for intersection in self.liu_intersections:
            MAPE_IO, MAPE_liu = intersection.get_total_MAPE_per_intersection()
            if MAPE_IO != 0 and MAPE_liu != 0:
                sum_MAPE_IO += MAPE_IO
                sum_MAPE_liu += MAPE_liu
                cnt += 1

        final_MAPE_IO = sum_MAPE_IO/cnt
        final_MAPE_liu = sum_MAPE_liu/cnt

        return final_MAPE_IO, final_MAPE_liu

    def get_total_hybrid_mape_veh(self):
        return np.mean([lane.get_hybrid_MAPE_veh() for lane in self.liu_lanes.values()])

    def get_total_standard_deviation_MAPE(self, final_MAPE_IO, final_MAPE_liu):
        sum_SD_IO = 0
        sum_SD_liu = 0
        n = 0
        for intersection in self.liu_intersections:
            SD_IO, SD_liu, num_lanes = intersection.get_sum_SD_per_intersection(final_MAPE_IO, final_MAPE_liu)
            sum_SD_IO += SD_IO
            sum_SD_liu += SD_liu
            n += num_lanes
        print('n:', n)
        SD_IO = math.sqrt(sum_SD_IO/n)
        SD_liu = math.sqrt(sum_SD_liu/n)

        print('SD_IO:', SD_IO)
        print('SD_liu:', SD_liu)

        return SD_IO, SD_liu, n



    def get_max_num_phase(self, end_time=np.inf):
        """Return the maximum number of phases for all LiuLanes in the network

        :param end_time: Upper time limit (optional), defaults to np.inf
        :return: Maximum number of phases
        :rtype: int
        """

        max_num_phase = max(
            [lane.num_queueing_intervals_before(end_time)
             for lane in self.liu_lanes.values()])
        max_num_phase = max_num_phase - 2
        # -2 because estimation for first and last phase is not possible

        max_phase_length = max(
            [i.get_max_phase_length() for i in self.liu_intersections], default=0)

        if np.isinf(end_time):
            end_time = 1e7
        max_num_phase_old = int((end_time/max_phase_length) - 2)


        if not max_num_phase == max_num_phase_old:
            _logger.debug('max_num_phase = %g, max_num_phase_old = %g',
                          max_num_phase, max_num_phase_old)

        return max_num_phase

    def store_results(self, unload_data = False):

        for intersection in self.liu_intersections:
            intersection.add_estimation_to_df(self.store_while_running, 0)

        if self.test_data:
            self.df_estimation_results.to_hdf(os.path.join(os.path.dirname(
                    self.sumo_network.netfile), 'liu_estimation_results_test_data_' + str(self.sim_num) + '.h5'),
                    key = 'df_estimation_results')
        else:
            self.df_estimation_results.to_hdf(os.path.join(os.path.dirname(
                    self.sumo_network.netfile), 'liu_estimation_results' + str(self.sim_num) + '.h5'),
                    key = 'df_estimation_results')
        print('Saved all results in hdf file')

        if unload_data == True: #unload data and delete everything to save memory
            for intersection in self.liu_intersections:
                intersection.unload_data()
                del intersection
            self.df_estimation_results = None
            print('Unload data from liu-method')

    def append_results(self, num_phase):

        if self.test_data:
            file_name = os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results_test_data_' + str(self.sim_num) + '.h5')
        else:
            file_name = os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results' + str(self.sim_num) + '.h5')
        if not os.path.exists(file_name) or num_phase == 0:
            self.df_estimation_results.to_hdf(file_name,
                    key = 'df_estimation_results', format='table')
            _logger.debug('Created new hdf file')
        else:
            #store = pd.HDFStore(os.path.join(os.path.dirname(self.sumo_network.netfile), 'liu_estimation_results.h5'))
            #ohlcv_candle.to_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile), 'liu_estimation_results.h5'))
            #store.append('df_estimation_results', self.df_estimation_results, format='t',  data_columns=True)
            self.df_estimation_results.to_hdf(file_name,
                    key = 'df_estimation_results', format='table', append = True)
            #print(store.info())
            #store.close()
            _logger.debug('Appended results in hdf file')

    def get_liu_lane_IDs(self):
        list_lane_names = []
        for lane in self.liu_lanes.values():
            list_lane_names.append(lane.lane_id)
        return list_lane_names

    def get_liu_results_path(self):
        return os.path.join(os.path.dirname(
            self.sumo_network.netfile), 'liu_estimation_results' + str(self.sim_num) + '.h5')


class LiuIntersection(object):
    # LiuLane objects corresponding to its component
    # lanes as well as intersection-level info like traffic light definition
    def __init__(self, sumolib_tls, parent, time_window):
        self.sumolib_tls = sumolib_tls
        self.parent = parent
        self.time_window = time_window

        self.liu_lanes = OrderedDict()
        for conn in self.sumolib_tls.getConnections():
            in_lane, out_lane, _ = conn
            # create a LiuLane for each in-lane
            # update this call if the instance lane needs more references to
            # eg dataframes
            in_lane_id = in_lane.getID()
            if self.parent.graph.nodes('detectors')[in_lane_id] is not None:
                if in_lane_id not in self.liu_lanes:
                    lane = LiuLane(in_lane, out_lane, self)
                    self.liu_lanes[in_lane_id] = lane
                else:
                    lane = self.liu_lanes[in_lane_id]
                lane.add_out_lane(out_lane.getID())

    def run_next_phase(self, num_phase):
        #run the single-phase calculation for each lane
        for lane in self.liu_lanes.values():
            lane.run_next_phase(num_phase)

    def plot_results(self, show_plot, show_infos):
        for lane in self.liu_lanes.values():
            lane.plot(show_plot, show_infos)

    def get_total_MAPE_per_intersection(self):
        sum_MAPE_IO = 0
        sum_MAPE_liu = 0
        cnt = 0
        for lane in self.liu_lanes.values():
            MAPE_IO, MAPE_liu, used = lane.get_MAPE()
            if used == True:
                sum_MAPE_IO += MAPE_IO
                sum_MAPE_liu += MAPE_liu
                cnt += 1
        if cnt > 0:
            sum_MAPE_IO = sum_MAPE_IO/cnt
            sum_MAPE_liu = sum_MAPE_liu/cnt
        return sum_MAPE_IO, sum_MAPE_liu

    def get_sum_SD_per_intersection(self, final_MAPE_IO, final_MAPE_liu):
        sum_SD_IO = 0
        sum_SD_liu = 0
        for lane in self.liu_lanes:
            MAPE_IO, MAPE_liu, used = lane.get_MAPE()
            if used == True:
                sum_SD_IO += (MAPE_IO - final_MAPE_IO)**2
                sum_SD_liu += (MAPE_liu - final_MAPE_liu)**2
        return sum_SD_IO, sum_SD_liu, len(self.liu_lanes)

    def get_max_phase_length(self):
        max_phase_length = max(
            [lane.get_phase_length() for lane in self.liu_lanes.values()],
            default=0)

        return max_phase_length

    def add_estimation_to_df(self, while_running, phase_cnt):
        for lane in self.liu_lanes.values():
            time, real_queue, estimated_queue, estimated_queue_pure_liu, phase_start, phase_end, tls_start, tls_end = lane.get_estimation_data(while_running)

            lane_ID = lane.lane_id
            iterables = [[lane_ID], ['time', 'ground-truth',
                                     'estimated hybrid',
                                     'estimated hybrid (veh)',
                                     'estimated pure liu', 'phase start',
                                     'phase end', 'tls start', 'tls end']]
            index = pd.MultiIndex.from_product(iterables, names=['lane', 'values'])
            if while_running == False:
                df_lane = pd.DataFrame(index = np.arange(1, len(time)+1), columns = index)
            else:
                df_lane = pd.DataFrame(index = [phase_cnt], columns = index)
            df_lane.index.name = 'phase'
            df_lane[lane_ID, 'time'] = time
            df_lane[lane_ID, 'ground-truth'] = real_queue
            df_lane[lane_ID, 'estimated hybrid'] = estimated_queue
            df_lane[lane_ID, 'estimated hybrid (veh)'] = estimated_queue * self.liu_lanes[lane_ID].k_j
            df_lane[lane_ID, 'estimated pure liu'] = estimated_queue_pure_liu
            df_lane[lane_ID, 'phase start'] = phase_start
            df_lane[lane_ID, 'phase end'] = phase_end
            df_lane[lane_ID, 'tls start'] = tls_start
            df_lane[lane_ID, 'tls end'] = tls_end

            self.parent.df_estimation_results = pd.concat([self.parent.df_estimation_results, df_lane], axis = 1)

    def unload_data(self):
        for lane in self.liu_lanes.values():
            lane.unload_data()
        self.liu_lanes = OrderedDict()

    def get_tls_output_filenames_for_lanes(self):
        return {lane_id: lane.tls_output_filename
                for lane_id, lane in self.liu_lanes.items()}


_DetInfo = namedtuple('det_info', ['id', 'info'])


_Queueing_Period_Data = namedtuple(
    'QueueingPeriodData',
    ['num_period', 'start_time', 'end_time',
     'list_time_stop', 'list_nVehContrib_stop',
     'list_time', 'list_occupancy', 'list_nVehEntered', 'list_nVehContrib',
     'list_time_e2', 'list_startedHalts_e2', 'list_max_jam_length_m_e2',
     'list_max_jam_length_veh_e2',
     'list_jam_length_sum_e2'])


class LiuLane(object):
    def __init__(self, sumolib_in_lane, out_lane, parent):
        # initialize this object: lane id, references to proper objects in the
        # big sumo_network file, and objects to store results
        # as well as tracithe state: last step/time period processed
        # (and leftover queue for if we need to use the input/output method in
        # any steps)
        self.reader = SumoLaneOutputReader(
            sumolib_in_lane, out_lane, parent.parent.reader,
            parent.parent.input_data_hdf_file)
        self.parent = parent

        self._init_result_arrays()

        self.L_d = float(self.graph.nodes('detectors')[self.lane_id][self.stopbar_detector_id]['pos']
            )-float(self.graph.nodes('detectors')[self.lane_id][self.adv_detector_id]['pos'])
        self.k_j = JAM_DENSITY #jam density
        self.v_2 = -6.5 #just for initialization

    def _init_result_arrays(self):
        self.arr_breakpoint_A = []
        self.arr_breakpoint_B = []
        self.arr_breakpoint_C = []
        self.arr_breakpoint_C_stopbar = []

        self.arr_estimated_max_queue_length = []
        self.arr_estimated_max_queue_length_pure_liu = []
        self.arr_estimated_time_max_queue = []
        self.used_method = []

    # Various properties that point to the reader's
    @property
    def lane_id(self):
        return self.reader.lane_id

    @property
    def sumolib_in_lane(self):
        return self.reader.sumolib_in_lane

    @property
    def sumolib_out_lane(self):
        return self.reader.sumolib_out_lane

    @property
    def out_lane_ids(self):
        return self.reader.out_lane_ids

    @property
    def tls_output_filename(self):
        return self.reader.tls_output_filename

    @property
    def graph(self):
        return self.parent.parent.graph

    @property
    def dash_lane_id(self):
        return self.reader.dash_lane_id

    @property
    def stopbar_detector_id(self):
        return self.reader.stopbar_detector_id

    @property
    def adv_detector_id(self):
        return self.reader.adv_detector_id

    @property
    def arr_real_max_queue_length(self):
        return self.reader.arr_real_max_queue_length

    @property
    def arr_maxJamLengthInMeters(self):
        return self.reader.arr_maxJamLengthInMeters

    @property
    def arr_maxJamLengthInVehicles(self):
        return self.reader.arr_maxJamLengthInVehicles

    @property
    def arr_maxJamLengthInVehiclesSum(self):
        return self.reader.arr_maxJamLengthInVehiclesSum

    @property
    def arr_phase_start(self):
        return self.reader.arr_phase_start

    @property
    def arr_phase_end(self):
        return self.reader.arr_phase_end

    @property
    def green_intervals(self):
        return self.reader.green_intervals

    @property
    def arr_green_phase_start(self):
        return self.reader.arr_green_phase_start

    @property
    def arr_green_phase_end(self):
        return self.reader.arr_green_phase_end

    @property
    def prev_cycle_parsed(self):
        return self.reader.prev_cycle_parsed

    @property
    def this_cycle_parsed(self):
        return self.reader.this_cycle_parsed

    @property
    def next_cycle_parsed(self):
        return self.reader.next_cycle_parsed

    @property
    def duration_green_light(self):
        return self.reader.duration_green_light

    @property
    def phase_length(self):
        return self.reader.phase_length

    @property
    def phase_start(self):
        return self.reader.phase_start

    def add_out_lane(self, out_lane_id):
        self.reader.add_out_lane(out_lane_id)

    def add_green_interval(self, start_time, end_time):
        self.reader.add_green_interval(start_time, end_time)

    def _init_max_per_green_estimate(self):
        # initialize value (empirical), but will be estimated during simulation run
        # 1 vehicle every 2 sec
        self.max_veh_leaving_on_green = int(0.5*np.diff(self.green_intervals[0]))
        max_veh_leaving_on_green_old = int(0.5*self.duration_green_light)
        assert max_veh_leaving_on_green_old == self.max_veh_leaving_on_green

    def nth_cycle_interval(self, n):
        """Returns the nth queueing and discharging time interval for this lane: (start, end)

        "start" corresponds to start of red phase (end of previous green phase).
        "end" corresponds to end of green phase

        :param n: Index of desired cycle time interval
        :type n: int
        :return: Start and end of cycle time interval
        :rtype: Tuple of ints
        """
        return self.reader.nth_cycle_interval(n)

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
        return self.reader.nth_green_phase_intervals(n)

    def nth_queueing_cycle_green_interval(self, n):
        assert n < len(self.green_intervals)
        assert n >= 0

        return self.green_intervals[n + 1]

    def _init_detector_parsers(self):
        self.reader._initialize_parser()

    def _estimate_fixed_cycle_timings(self):
        self.reader._estimate_fixed_cycle_timings()

    def get_tls_output_filename(self):
        """Return the filename of the tls-switch output file from the nx graph.

        :return: tls-switch filename
        :rtype: string
        """
        return self.reader.get_tls_output_filename()

    def parse_cycle_data(self, num_phase):
        return self.reader.parse_cycle_data(num_phase)

    def run_next_phase(self, num_phase):
        """Run the Liu method for the given phase.

        :param num_phase: phase index
        :type num_phase: int
        """
        if num_phase >= self.num_queueing_intervals() - 1:
            return

        start, end, curr_e1_stopbar, curr_e1_adv_detector, curr_e2_detector = self.parse_cycle_data(num_phase)
        self.breakpoint_identification(num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector)
        self.C_identification_short_queue(num_phase, start, end, curr_e1_stopbar)
        self.queue_estimate(num_phase, start, end, curr_e1_adv_detector, curr_e1_stopbar)
        self.get_ground_truth_queue(num_phase, start, end, curr_e2_detector)

    def breakpoint_identification(self, num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector):

        # Calculate binary occupancy
        # binary occupancy = 1 iff a car is on the detector for a full sim step (meaning it is stopped)
        binary_occ_t = (curr_e1_adv_detector['occupancy'] >= 100).astype(np.bool)

        # Calculate the time gap between vehicles
        # Assume that there is maximum only one vehicle per second on the detector
        point_of_time = []
        time_gap_vehicles = []
        time_cnt = 0

        next_red = self.nth_cycle_interval(num_phase + 1)[1]

        if not next_red == end + self.phase_length:
            _logger.info('lane %s, next_red = %g, end + self.phase_length = %g',
                         self.lane_id, next_red, end + self.phase_length)

        try:
            for t in range(start, next_red):
                if curr_e1_adv_detector['nVehEntered'][t] == 1: #new vehicle enters the detector: start timer new timer and save old measurements
                    time_gap_vehicles.append(time_cnt)
                    point_of_time.append(t)
                    time_cnt = 0 #reset time counter for new timing

                if curr_e1_adv_detector['nVehEntered'][t] == 0: #timer is running, following vehicle hasn't come yet
                    time_cnt = time_cnt+1
        except KeyError:
            _logger.info(
                'lane_id = %s' +
                'start = %s, end = %s, next_red = %s, phase_length = %s, nth_cycle_interval = %s'
                + ' n+1th_cycle_interval = %s',
                self.lane_id, start, end, next_red, self.phase_length,
                self.nth_cycle_interval(num_phase), self.nth_cycle_interval(num_phase+1))
            raise


        ### Characterize the Breakpoints A,B ###
        ### A & B ### use the binary occupancy:

        # reverse the series, then see if detector is fully occupied for the
        # next 4 seconds
        breakpoint_A = (binary_occ_t[::-1].rolling(4)
                                          .agg(lambda x: x.all())[::-1]
                                          .fillna(0)
                                          .astype(np.bool))

        # see if the detector is fully occupied for this timestep and the
        # previous 3 seconds
        breakpoint_B = (binary_occ_t.rolling(4)
                                    .agg(lambda x: x.all())
                                    .astype(np.bool))

        bool_A_found = False
        bool_B_found = False
        breakpoint_A = False
        breakpoint_B = False

        for t in range(start, end):

            if not bool_A_found and not binary_occ_t[t] and binary_occ_t[t+1] and binary_occ_t[t+2] and binary_occ_t[t+3]:
                breakpoint_A = t
                bool_A_found = True

            if bool_A_found and not bool_B_found and binary_occ_t[t-3] and binary_occ_t[t-2] and binary_occ_t[t-1] and not binary_occ_t[t]:
                breakpoint_B = t
                bool_B_found = True

        if bool_A_found and bool_B_found:
            self.arr_breakpoint_A.append(breakpoint_A)  #store breakpoints
            self.arr_breakpoint_B.append(breakpoint_B)  #store breakpoints

            #estimating how many vehicles can leave the lane during green phase! (for each lane)
            max_veh_leaving = sum(curr_e1_stopbar["nVehContrib"][start:end])
            if max_veh_leaving > self.max_veh_leaving_on_green:
                self.max_veh_leaving_on_green = max_veh_leaving

        else:
            self.arr_breakpoint_A.append(-1)  #store breakpoints
            self.arr_breakpoint_B.append(-1)  #store breakpoints

        ### Characterizing Breakpoint C ### using time gap between consecutive vehicles
        bool_C_found = False
        breakpoint_C = 0
        start_search = breakpoint_B + 10 #start searching for C after the breakpoint B + 10 seconds and until end; little offset of 10 sec is necessary to avoid influence from breakpoint B
        end_search = end + 50

    ###ATTENTION!! Breakpoint k-1 chosen!!! (alternative k, but that is overestimating!)
        for k in range(0, len(point_of_time)-1):
            if (point_of_time[k] >= start_search
                and point_of_time[k] <= end_search
                and time_gap_vehicles[k] >= 4
                and time_gap_vehicles[k] >= time_gap_vehicles[k-1]
                and point_of_time[k-1] >= breakpoint_B
                and not bool_C_found and bool_B_found
            ):

                breakpoint_C = point_of_time[k-1]
                bool_C_found = True
                self.arr_breakpoint_C.append(breakpoint_C)  #store breakpoints

        if not bool_C_found:
            self.arr_breakpoint_C.append(-1)  #store breakpoints

    def C_identification_short_queue(self, num_phase, start, end, curr_e1_stopbar):
        #Calculating the time gap between vehicles
        #I am assuming that there is maximum only one vehicle per second on the detector
        point_of_time = []
        time_gap_vehicles = []
        time_cnt = 0

        next_red = self.nth_cycle_interval(num_phase + 1)[1]
        for t in range(start, next_red):
            if curr_e1_stopbar['nVehContrib'][t] == 1: #new vehicle enters the detector: start timer new timer and save old measurements
                time_gap_vehicles.append(time_cnt)
                point_of_time.append(t)
                time_cnt=0 #reset time counter for new timing

            if curr_e1_stopbar['nVehContrib'][t] == 0: #timer is running, following vehicle hasn't come yet
                time_cnt= time_cnt+1

        ### Characterizing Breakpoint C ### using time gap between consecutive vehicles
        bool_C_found = False
        breakpoint_C = 0
        start_search = end-self.duration_green_light + 2 #start searching for C after the green start + 2 seconds and until end
        end_search = end

    ###ATTENTION!! Breakpoint k-1 chosen!!! (alternative k, but that is overestimating!)
        for k in range(0, len(point_of_time)-1):
            if point_of_time[k] >= start_search and point_of_time[k] <= end_search and time_gap_vehicles[k] >= 4 and time_gap_vehicles[k] >= time_gap_vehicles[k-1]:

                breakpoint_C = point_of_time[k-1]
                bool_C_found = True
                self.arr_breakpoint_C_stopbar.append(breakpoint_C)  #store breakpoints

        if not bool_C_found:
            self.arr_breakpoint_C_stopbar.append(-1)  #store breakpoints


    def queue_estimate(self, num_phase, start, end, curr_e1_adv_detector, curr_e1_stopbar):

        self.arr_phase_start.append(start)
        self.arr_phase_end.append(end)
        green_phase_start, green_phase_end = self.nth_green_phase_intervals(num_phase)
        self.arr_green_phase_start.append(green_phase_start)
        self.arr_green_phase_end.append(green_phase_end)
        #check if breakpoint A exists
        if self.arr_breakpoint_A[-1] == -1:


            #simple input-output method
            if len(self.arr_estimated_max_queue_length) == 0:
                old_estimated_queue_nVeh = 0
            else:
                old_estimated_queue_nVeh = self.arr_estimated_max_queue_length[-1]*self.k_j

            speed = self.sumolib_in_lane.getSpeed()
            time_gap = int(self.L_d/speed)
            estimated_queue_nVeh = max(
                    old_estimated_queue_nVeh - self.max_veh_leaving_on_green
                    + sum(curr_e1_adv_detector["nVehContrib"][start-self.duration_green_light:start-time_gap]),
                0) + sum(curr_e1_adv_detector["nVehContrib"][start-time_gap:end-self.duration_green_light])
            self.arr_estimated_max_queue_length.append(estimated_queue_nVeh/self.k_j)
            self.used_method.append(0)

            breakpoint_C_stopbar = self.arr_breakpoint_C_stopbar[len(self.arr_breakpoint_C_stopbar)-1]
            ### Expansion I for short queue
            #n is the number of vehicles passing detector between T_ng(start red phase) and T_C (breakpoint C)

            if self.arr_breakpoint_C_stopbar[len(self.arr_breakpoint_C_stopbar)-1] == -1:
                n = sum(curr_e1_stopbar["nVehContrib"].loc[(end-self.duration_green_light):end])
            else:
                n = sum(curr_e1_stopbar["nVehContrib"].loc[(end-self.duration_green_light):breakpoint_C_stopbar])

            L_max = n/self.k_j

            self.arr_estimated_max_queue_length_pure_liu.append(L_max)
            self.arr_estimated_time_max_queue.append(end-self.duration_green_light)


        else:
            breakpoint_B = self.arr_breakpoint_B[len(self.arr_breakpoint_B)-1]

            if self.arr_breakpoint_C[len(self.arr_breakpoint_C)-1] == -1 or self.arr_breakpoint_C[len(self.arr_breakpoint_A)-2] == self.arr_breakpoint_A[len(self.arr_breakpoint_A)-1]:
                if len(self.arr_estimated_max_queue_length) == 0:
                    self.arr_estimated_max_queue_length.append(self.L_d)
                else:
                    self.arr_estimated_max_queue_length.append(self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1])
                self.arr_estimated_time_max_queue.append(end-self.duration_green_light)
                self.used_method.append(2)
            else:
                breakpoint_C = self.arr_breakpoint_C[len(self.arr_breakpoint_C)-1]
                try:
                    v_2 = self.L_d/(breakpoint_B-(end-self.duration_green_light))
                    self.v_2 = -abs(v_2)
                except ZeroDivisionError: # if breakpoint_B is at the beginning of the green light
                    pass

                ### Expansion I
                #n is the number of vehicles passing detector between T_ng(start red phase) and T_C (breakpoint C)
                n = sum(curr_e1_adv_detector["nVehEntered"].loc[(end-self.duration_green_light):breakpoint_C])
                L_max = n/self.k_j + self.L_d
                #T_max = (end-self.duration_green_light) + L_max/abs(self.v_2)

                self.arr_estimated_max_queue_length.append(L_max)
                #self.arr_estimated_time_max_queue.append(T_max)
                self.arr_estimated_time_max_queue.append(end-self.duration_green_light)
                self.used_method.append(1)

            self.arr_estimated_max_queue_length_pure_liu.append(
                    self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1])
        # next iteration of queue estimation from breakpoints

    def get_last_queue_estimate(self):
        # return last-estimated queue
        return (self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1],
                self.arr_estimated_time_max_queue[len(self.arr_estimated_time_max_queue)-1])

    def get_queue_estimate(self):
        # return  whole estimated queue until now
        return self.arr_estimated_max_queue_length, self.arr_estimated_time_max_queue

    def get_ground_truth_queue(self, num_phase, start, end, curr_e2_detector):
        #calculate ground truth data for queue and store in self arrays
        self.arr_real_max_queue_length.append(
                sum(curr_e2_detector['startedHalts'][start:end])/self.k_j)
        self.arr_maxJamLengthInMeters.append(
                max(np.array(curr_e2_detector['maxJamLengthInMeters'][start:end])))
        self.arr_maxJamLengthInVehicles.append(
                max(np.array(curr_e2_detector['maxJamLengthInVehicles'][start:end])))
        return sum(curr_e2_detector['startedHalts'][start:end])/self.k_j

    def get_MAPE(self):

        #calculating MAPE for liu + IO
        sum_mape_IO = 0
        cnt = 0

        if self.parent.parent.use_started_halts == True:
            arr_real_queue = self.arr_real_max_queue_length
        else:
            arr_real_queue = self.arr_maxJamLengthInMeters

        if len(self.arr_estimated_max_queue_length) > 2:
            for estimation_queue, real_queue in zip(self.arr_estimated_max_queue_length, arr_real_queue):
                if estimation_queue != 0 and real_queue != 0:
                    sum_mape_IO = sum_mape_IO + abs((real_queue-estimation_queue)/real_queue)
                    cnt += 1

            if cnt > 0:
                sum_mape_IO = sum_mape_IO/cnt

            #calculating MAPE for pure liu
            sum_mape_liu = 0
            cnt = 0

            for estimation_queue, real_queue in zip(self.arr_estimated_max_queue_length_pure_liu, arr_real_queue):
                if estimation_queue != 0 and real_queue != 0:
                    sum_mape_liu = sum_mape_liu + abs((real_queue-estimation_queue)/real_queue)
                    cnt += 1
            if cnt > 0:
                sum_mape_liu = sum_mape_liu/cnt

            # in some cases the MAPE is 0 because the estimation is perfect
            # we have to count this cases into the MAPE; check if lane is used
            if sum(arr_real_queue) >0:
                used = True
            else:
                used = False

            return sum_mape_IO, sum_mape_liu, used
        else:
            return -1, -1

    def get_hybrid_MAPE_meters(self):
        if self.parent.parent.use_started_halts:
            true = np.array(self.arr_real_max_queue_length)
        else:
            true = np.array(self.arr_maxJamLengthInMeters)
        est = np.array(self.arr_estimated_max_queue_length)

        return self._mape_from_lists(true, est)

    def _mape_from_lists(self, true, est):
        mape = np.zeros_like(true)
        where = np.nonzero(true)
        mape[where] = np.abs(true[where] - est[where]) / true[where]
        mape[not(where) and est != 0] = 1.

        return np.mean(mape)

    def get_hybrid_MAPE_veh(self):
        if self.parent.parent.use_started_halts:
            true = np.array(self.arr_real_max_queue_length) * self.k_j
        else:
            true = np.array(self.arr_maxJamLengthInVehicles)
        est = np.array(self.arr_estimated_max_queue_length) * self.k_j

        return self._mape_from_lists(true, est)

    def plot(self, show_graph, show_infos):
        from matplotlib import pyplot as plt
        if show_graph == True:
            start = 0
            fig = plt.figure()
            fig.set_figheight(5)
            fig.set_figwidth(12)

            estimation, = plt.plot(self.arr_estimated_time_max_queue, self.arr_estimated_max_queue_length, c='r', label= 'hybrid model')
            # ground_truth, = plt.plot(self.arr_estimated_time_max_queue, self.arr_real_max_queue_length, c='b', label= 'sum started halts')
            # estimation_pure_liu, = plt.plot(self.arr_estimated_time_max_queue, self.arr_estimated_max_queue_length_pure_liu, c='m', label= 'expansion model', linestyle='--')
            max_length_queue, = plt.plot(self.arr_estimated_time_max_queue, self.arr_maxJamLengthInMeters, c='k', label= 'maxJamLength e2 detector', linestyle='--')

            y_lim = max(map(max, self.arr_estimated_max_queue_length, self.arr_maxJamLengthInMeters))
            x_lim = max(self.arr_estimated_time_max_queue)

            plt.legend(handles=[
                estimation,
                # ground_truth,
                # estimation_pure_liu,
                max_length_queue], fontsize = 18)

            plt.xticks(np.arange(0, 6000, 250))
            plt.xticks(fontsize=18)
            plt.yticks(np.arange(0, y_lim, 50))
            plt.yticks(fontsize=18)
            plt.xlim(0, x_lim)
            plt.ylim(0, y_lim)
            if self.sumolib_in_lane.getID()== 'bottom2to2/0_2':
                plt.ylim(0, 700)
            plt.xlabel('time [s]', fontsize = 18)
            plt.ylabel('queue length [m]', fontsize = 18)

            for i in range(len(self.arr_estimated_time_max_queue)):
                start = start+self.phase_length #MODIFIED -> shift of one phase!
                end = start + self.phase_length #seconds #end is end of green phase
                if self.used_method[i] ==0:
                    plt.axvspan(start, end, alpha=0.5, color='green') #'simple input output method'
                elif self.used_method[i] == 1:
                    plt.axvspan(start, end, alpha=0.5, color='yellow') #'Liu method'
                elif self.used_method[i] == 2:
                    plt.axvspan(start, end, alpha=0.5, color='red') #'oversaturation'

            plt.show()

            #just to save plots, delete after writing
            if self.sumolib_in_lane.getID()== 'bottom2to2/0_2':
                fig.savefig("simonnet_"+self.dash_lane_id+".pdf", bbox_inches='tight')
#            if self.sumolib_in_lane.getID()== '1/0to1/1_0' or self.sumolib_in_lane.getID()== '1/0to1/1_1' or self.sumolib_in_lane.getID()== '1/0to1/1_2':
#                fig.savefig("simonnet_"+dash_lane_id+".pdf", bbox_inches='tight')

        if show_infos == True:
            #show some stats for debug
            print('lane id:', self.sumolib_in_lane.getID())
            print('out lane id:', self.sumolib_out_lane.getID())
            print('Estimated queue length: ', self.arr_estimated_max_queue_length)
            print('real queue length: ', self.arr_real_max_queue_length)
            print('e2 detector maxJamLengthInMeters: ', self.arr_maxJamLengthInMeters)
            print('phase length:', self.phase_length)
            print('phase start:', self.phase_start)
            print('-----------')

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

    def num_queueing_intervals(self):
        """Get the number of queueing intervals (red to red) from tls switch data
        """
        counter = 0
        while True:
            try:
                self.nth_cycle_interval(counter)
            except IndexError:
                return counter
            counter += 1

    def num_queueing_intervals_before(self, end_time):
        counter = 0
        while True:
            try:
                interval = self.nth_cycle_interval(counter)
                assert interval[0] <= end_time
            except (IndexError, AssertionError):
                return counter
            counter += 1

    def get_estimation_data(self, while_running):
        if while_running == True:
            return(self.arr_estimated_time_max_queue[len(self.arr_estimated_time_max_queue)-1],
                   self.arr_real_max_queue_length[len(self.arr_real_max_queue_length)-1],
                   self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1],
                   self.arr_estimated_max_queue_length_pure_liu[len(self.arr_estimated_max_queue_length_pure_liu)-1],
                   self.arr_phase_start[len(self.arr_phase_start)-1],
                   self.arr_phase_end[len(self.arr_phase_end)-1],
                   self.arr_green_phase_start[len(self.arr_green_phase_start)-1],
                   self.arr_green_phase_end[len(self.arr_green_phase_end)-1])

        else:
            return (self.arr_estimated_time_max_queue, self.arr_real_max_queue_length,
                    self.arr_estimated_max_queue_length,
                    self.arr_estimated_max_queue_length_pure_liu,
                    self.arr_phase_start,
                    self.arr_phase_end,
                    self.arr_green_phase_start,
                    self.arr_green_phase_end)

    def get_lane_ID(self):
        return self.sumolib_in_lane.getID()

    def unload_data(self):
        self.parsed_xml_e1_stopbar_detector = None
        self.parsed_xml_e1_adv_detector = None
        self.parsed_xml_e2_detector = None
        self._init_result_arrays()
