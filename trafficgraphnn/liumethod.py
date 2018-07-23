import logging
import os
import sys
import xml.etree.cElementTree as et

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from trafficgraphnn.utils import iterfy
from trafficgraphnn.get_tls_data import get_tls_data


_logger = logging.getLogger(__name__)

JAM_DENSITY = 0.13333  # hardcoded jam density value (veh/meter)

class LiuEtAlRunner(object):
    def __init__(self, sumo_network, lane_subset=None, time_window=None, store_while_running = True):
        self.sumo_network = sumo_network
        self.graph = self.sumo_network.get_graph()
        self.net = sumo_network.net
        self.parsed_xml_tls = None
        self.df_estimation_results = pd.DataFrame()
        self.store_while_running = store_while_running
        # verify all lanes requested are actually in the network
        if lane_subset is None:
            # process all lanes
            self.lane_subset = self.sumo_network.graph.nodes
        else:
            lane_subset = iterfy(lane_subset)
            lanes_not_in_graph = [
                lane for lane in lane_subset
                if lane not in self.sumo_network.nodes
            ]
            if len(lanes_not_in_graph) > 0:
                _logger.warning(
                    'The following lanes are not present in the supplied'
                    ' network and will be ignored: {}'.format(
                        lanes_not_in_graph)
                )
                self.lane_subset = [
                    lane for lane in lane_subset
                    if lane in self.sumo_network.nodes
                ]

        # create Intersection objects for each intersection
        self.liu_intersections = []
        for tls in self.net.getTrafficLights():
            intersection = LiuIntersection(tls, self, time_window=time_window)
            self.liu_intersections.append(intersection)

    def run_up_to_phase(self, max_num_phase):
        # iterate on the single-step methods for each intersection until
        # reaching the given time
        for num_phase in range(1, max_num_phase):
            print('Running estimation for every lane in every intersection in phase', num_phase)
            self.df_estimation_results = pd.DataFrame() #reset df for case of online storage
            for intersection in self.liu_intersections:
                intersection.run_next_phase(num_phase)
                if self.store_while_running == True:
                    intersection.add_estimation_to_df(self.store_while_running, num_phase)
            if self.store_while_running == True:
                self.append_results(num_phase)

        if self.store_while_running == False:
            self.store_results(unload_data = True)

        pass

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

    def get_max_num_phase(self, end_time):
        max_phase_length = 0
        for intersection in self.liu_intersections:
            if intersection.get_max_phase_length() > max_phase_length:
                max_phase_length = intersection.get_max_phase_length()

        max_num_phase = int((end_time/max_phase_length) -2)
        #-2 because estimation for first and last phase is not possible

        return max_num_phase

    def store_results(self, unload_data = False):

        for intersection in self.liu_intersections:
            intersection.add_estimation_to_df(self.store_while_running, 0)

        self.df_estimation_results.to_hdf(os.path.join(os.path.dirname(
                self.sumo_network.netfile), 'liu_estimation_results.h5'), key = 'df_estimation_results')
        print('Saved all results in hdf file')

        if unload_data == True: #unload data and delete everything to save memory
            for intersection in self.liu_intersections:
                intersection.unload_data()
                del intersection
            self.df_estimation_results = None
            print('Unload data from liu-method')
        pass

    def append_results(self, num_phase):
       
        if num_phase ==1: 
            self.df_estimation_results.to_hdf(os.path.join(os.path.dirname(
                    self.sumo_network.netfile), 'liu_estimation_results.h5'),
                    key = 'df_estimation_results', format='table')
            print('Created new hdf file')
        else:
            #store = pd.HDFStore(os.path.join(os.path.dirname(self.sumo_network.netfile), 'liu_estimation_results.h5'))
            #ohlcv_candle.to_hdf(os.path.join(os.path.dirname(self.sumo_network.netfile), 'liu_estimation_results.h5'))
            #store.append('df_estimation_results', self.df_estimation_results, format='t',  data_columns=True)
            self.df_estimation_results.to_hdf(os.path.join(os.path.dirname(
                    self.sumo_network.netfile), 'liu_estimation_results.h5'),
                    key = 'df_estimation_results', format='table', append = True)
            #print(store.info())
            #store.close()
            print('Appended results in hdf file')

        pass


class LiuIntersection(object):
    # LiuLane objects corresponding to its component
    # lanes as well as intersection-level info like traffic light definition
    def __init__(self, sumolib_tls, parent, time_window):
        self.sumolib_tls = sumolib_tls
        self.parent = parent
        self.time_window = time_window

        self.liu_lanes = []
        liu_lanes_id = []
        for conn in self.sumolib_tls.getConnections():
            in_lane, out_lane, conn_id = conn
            # create a LiuLane for each in-lane
            # update this call if the instance lane needs more references to
            # eg dataframes
            in_lane_id = in_lane.getID()
            if (
                in_lane_id not in liu_lanes_id
                and self.parent.graph.nodes('detectors')[in_lane.getID()] is not None
            ):
                liu_lanes_id.append(in_lane_id)
                #print('Creating Liu lane with in_lane = ', in_lane_id, 'and out_lane = ', out_lane_id)
                lane = LiuLane(in_lane, out_lane, self)
                self.liu_lanes.append(lane)


    def run_next_phase(self, num_phase):
        #run the single-phase calculation for each lane
        for lane in self.liu_lanes:
            lane.run_next_phase(num_phase)

    def plot_results(self, show_plot, show_infos):
        for lane in self.liu_lanes:
            lane.plot(show_plot, show_infos)
            print('MAPE: ', lane.get_MAPE())

    def get_total_MAPE_per_intersection(self):
        sum_MAPE_IO = 0
        sum_MAPE_liu = 0
        cnt = 0
        for lane in self.liu_lanes:
            MAPE_IO, MAPE_liu, used = lane.get_MAPE()
            if used == True:
                sum_MAPE_IO += MAPE_IO
                sum_MAPE_liu += MAPE_liu
                cnt += 1
        if cnt > 0:
            sum_MAPE_IO = sum_MAPE_IO/cnt
            sum_MAPE_liu = sum_MAPE_liu/cnt
        return sum_MAPE_IO, sum_MAPE_liu

    def get_max_phase_length(self):
        max_phase_length = 0
        for current_lane in self.liu_lanes:
            if current_lane.get_phase_length() > max_phase_length:
                max_phase_length = current_lane.get_phase_length()

        return max_phase_length

    def add_estimation_to_df(self, while_running, phase_cnt):
        for lane in self.liu_lanes:
            time, real_queue, estimated_queue, estimated_queue_pure_liu, phase_start, phase_end = lane.get_estimation_data(while_running)
            
            lane_ID = lane.get_lane_ID()
            iterables = [[lane_ID], ['time', 'ground-truth', 'estimated hybrid', 'estimated pure liu', 'phase start', 'phase end']]
            index = pd.MultiIndex.from_product(iterables, names=['lane', 'values'])
            if while_running == False:
                df_lane = pd.DataFrame(index = np.arange(1, len(time)+1), columns = index)
            else:
                df_lane = pd.DataFrame(index = [phase_cnt], columns = index)
            df_lane.index.name = 'phase'
            df_lane[lane_ID, 'time'] = time
            df_lane[lane_ID, 'ground-truth'] = real_queue
            df_lane[lane_ID, 'estimated hybrid'] = estimated_queue
            df_lane[lane_ID, 'estimated pure liu'] = estimated_queue_pure_liu
            df_lane[lane_ID, 'phase start'] = phase_start
            df_lane[lane_ID, 'phase end'] = phase_end
            
            self.parent.df_estimation_results = pd.concat([self.parent.df_estimation_results, df_lane], axis = 1)       
        pass

    def unload_data(self):
        for lane in self.liu_lanes:
            lane.unload_data()
            del lane


class LiuLane(object):
    def __init__(self, sumolib_in_lane, out_lane, parent):
        # initialize this object: lane id, references to proper objects in the
        # big sumo_network file, and objects to store results
        # as well as tracithe state: last step/time period processed
        # (and leftover queue for if we need to use the input/output method in
        # any steps)
        self.sumolib_in_lane = sumolib_in_lane
        self.sumolib_out_lane = out_lane

        #dataframes that are loaded once at the beginning
        self.df_e1_adv_detector = pd.DataFrame()
        self.df_e1_stopbar = pd.DataFrame()
        self.df_e2_detector = pd.DataFrame()
        self.df_traffic_lights = pd.DataFrame()
        self.graph = parent.parent.graph

        columns_e1_adv = ['time', 'occupancy', 'nVehEntered', 'nVehContrib']
        columns_e1_stopbar = ['time', 'nVehContrib']
        self.curr_e1_adv_detector = pd.DataFrame(columns=columns_e1_adv)
        self.curr_e1_stopbar = pd.DataFrame(columns=columns_e1_stopbar)
        columns_e2 = ['time', 'startedHalts']
        self.curr_e2_detector = pd.DataFrame(columns=columns_e2)

        self.parsed_xml_e1_stopbar_detector = None
        self.parsed_xml_e1_adv_detector = None
        self.parsed_xml_e2_detector = None


        #arrays for store results
        self.arr_breakpoint_A = []
        self.arr_breakpoint_B = []
        self.arr_breakpoint_C = []
        self.arr_breakpoint_C_stopbar = []

        self.arr_estimated_max_queue_length = []
        self.arr_estimated_max_queue_length_pure_liu = []
        self.arr_estimated_time_max_queue = []
        self.arr_real_max_queue_length = []
        self.used_method = []
        self.arr_phase_start = []
        self.arr_phase_end = []

        #parameters for lane
        lane_id = self.sumolib_in_lane.getID()
        dash_lane_id = lane_id.replace('/', '-')
        adv_detector_id = "e1_" + dash_lane_id + "_1"
        stopbar_detector_id = "e1_" + dash_lane_id + "_0"

        out_lane_id = self.sumolib_out_lane.getID()

        self.L_d = float(self.graph.nodes('detectors')[self.sumolib_in_lane.getID()][stopbar_detector_id]['pos']
            )-float(self.graph.nodes('detectors')[self.sumolib_in_lane.getID()][adv_detector_id]['pos'])
        self.k_j = JAM_DENSITY #jam density
        self.v_2 = -6.5 #just for initialization

        self.parent = parent
        #self.time_window = time_window

        #estimating tls data!
        if self.parent.parent.parsed_xml_tls == None:
            self.parent.parent.parsed_xml_tls = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                        self.graph.edges[lane_id, out_lane_id]['tls_output_info']['dest']))
            
            (self.phase_start, self.phase_length, self.duration_green_light
                     ) = get_tls_data(self.parent.parent.parsed_xml_tls, lane_id)


        self.parent.parent.parsed_xml_tls = None

        # initialize value (empirical), but will be estimated during simulation run
        self.max_veh_leaving_on_green = int(0.5*self.duration_green_light)


        # we would like to be able to have each lane's calculation of its Liu
        # state be entirely local to its instance of this class

    def run_next_phase(self, num_phase):
        # run one iteration of the liu method for the next phase in the
        # data history
        start, end, curr_e1_stopbar, curr_e1_adv_detector, curr_e2_detector = self.parse_cycle_data(num_phase)
        self.breakpoint_identification(num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector)
        self.C_identification_short_queue(start, end, curr_e1_stopbar)
        self.queue_estimate(num_phase, start, end, curr_e1_adv_detector, curr_e1_stopbar)
        self.get_ground_truth_queue(num_phase, start, end, curr_e2_detector)

    def next_phase_ready_to_be_run(self):
        # return True if we can run liu for the next as-yet-unrun phase and
        # False otherwise: use to make sure we don't get ahead of ourselves
        # when running in real time
        pass

    def parse_cycle_data(self, num_phase):
        #parse the data from the dataframes and write to the arrays in every cycle

        lane_id = self.sumolib_in_lane.getID()
        dash_lane_id = lane_id.replace('/', '-')
        adv_detector_id = "e1_" + dash_lane_id + "_1"
        stopbar_detector_id = "e1_" + dash_lane_id + "_0"

        start = int(self.phase_start + num_phase*self.phase_length) #seconds #start begins with red phase
        end = start + self.phase_length #seconds #end is end of green phase

        #using the last THREE phases, because they are needed to estimate breakpoints from the second last one!
        #one in future for Berakpoint C, one in past for simple input-output method

        ###  parse e1 andvanced and stopbar detector from xml file ####

        if self.parsed_xml_e1_stopbar_detector == None:
            self.parsed_xml_e1_stopbar_detector = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                                 self.graph.nodes('detectors')[lane_id][stopbar_detector_id]['file']))

        list_time_stop = []     #for stop bar detector
        list_nVehContrib_stop = []

        for node in self.parsed_xml_e1_stopbar_detector.getroot():
            begin = float(node.attrib.get('begin'))
            det_id = node.attrib.get('id')
            if begin >= start-self.phase_length and begin < end+self.phase_length and det_id == stopbar_detector_id:
                list_time_stop.append(begin)
                list_nVehContrib_stop.append(float(node.attrib.get('nVehContrib')))

        self.curr_e1_stopbar = pd.DataFrame(
                {'time': list_time_stop, 'nVehContrib': list_nVehContrib_stop})

        if self.parsed_xml_e1_adv_detector == None:
            self.parsed_xml_e1_adv_detector = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                                 self.graph.nodes('detectors')[lane_id][adv_detector_id]['file']))

        list_time = []       #for advanced detector
        list_occupancy = []
        list_nVehEntered = []
        list_nVehContrib = []
        for node in self.parsed_xml_e1_adv_detector.getroot():
            begin = float(node.attrib.get('begin'))
            det_id = node.attrib.get('id')
            if begin >= start-self.phase_length and begin < end+self.phase_length and det_id == adv_detector_id:
                list_time.append(begin)
                list_occupancy.append(float(node.attrib.get('occupancy')))
                list_nVehEntered.append(float(node.attrib.get('nVehEntered')))
                list_nVehContrib.append(float(node.attrib.get('nVehContrib')))

        self.curr_e1_adv_detector = pd.DataFrame(
                {'time': list_time, 'occupancy': list_occupancy,
                 'nVehEntered': list_nVehEntered, 'nVehContrib': list_nVehContrib})

        self.curr_e1_adv_detector.set_index('time', inplace=True)
        self.curr_e1_stopbar.set_index('time', inplace=True)

        ### parse e2 detector from xml file ###

        e2_detector_id = "e2_" + dash_lane_id + "_0"


        if self.parsed_xml_e2_detector == None:
            self.parsed_xml_e2_detector = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                                 self.graph.nodes('detectors')[lane_id][e2_detector_id]['file']))

        list_time_e2 = []
        list_startedHalts_e2 = []

        for node in self.parsed_xml_e2_detector.getroot():
            begin = float(node.attrib.get('begin'))
            det_id = node.attrib.get('id')
            if begin >= start and begin < end+self.phase_length and det_id == e2_detector_id:
                list_time_e2.append(begin)
                list_startedHalts_e2.append(float(node.attrib.get('startedHalts')))

        self.curr_e2_detector = pd.DataFrame(
                {'time': list_time_e2, 'startedHalts': list_startedHalts_e2})
        self.curr_e2_detector.set_index('time', inplace=True)


        return start, end, self.curr_e1_stopbar, self.curr_e1_adv_detector, self.curr_e2_detector

    def breakpoint_identification(self, num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector):

        #Calculate binary occupancy
        binary_occ_t = pd.Series()
        binary_occ_t = curr_e1_adv_detector['occupancy'].apply(lambda x: (1 if x >= 100 else 0))

        #Calculating the time gap between vehicles
        #I am assuming that there is maximum only one vehicle per second on the detector
        point_of_time = []
        time_gap_vehicles = []
        time_cnt = 0

        for t in range(start, end+self.phase_length):
            if curr_e1_adv_detector['nVehEntered'][t] == 1: #new vehicle enters the detector: start timer new timer and save old measurements
                time_gap_vehicles.append(time_cnt)
                point_of_time.append(t)
                time_cnt = 0 #reset time counter for new timing

            if curr_e1_adv_detector['nVehEntered'][t] == 0: #timer is running, following vehicle hasn't come yet
                time_cnt = time_cnt+1


        ### Characterize the Breakpoints A,B ###
        ### A & B ### use the binary occupancy:
        bool_A_found = 0
        bool_B_found = 0
        breakpoint_A = 0
        breakpoint_B = 0

        for t in range(start, end):

            if bool_A_found == 0 and binary_occ_t[t] == 0 and binary_occ_t[t+1] == 1 and binary_occ_t[t+2] == 1 and binary_occ_t[t+3] == 1:
                breakpoint_A = t
                bool_A_found = 1

            if bool_A_found == 1 and bool_B_found == 0 and binary_occ_t[t-3] == 1 and binary_occ_t[t-2] == 1 and binary_occ_t[t-1] == 1 and binary_occ_t[t] == 0:
                breakpoint_B = t
                bool_B_found = 1

        if bool_A_found == 1 and bool_B_found == 1:
            self.arr_breakpoint_A.append(breakpoint_A)  #store breakpoints
            self.arr_breakpoint_B.append(breakpoint_B)  #store breakpoints

            #estimating how many vehicles can leave the lane during green phase! (for each lane)
            max_veh_leaving = sum(curr_e1_stopbar["nVehContrib"][end-self.duration_green_light:end])
            if max_veh_leaving > self.max_veh_leaving_on_green:
                self.max_veh_leaving_on_green = max_veh_leaving

        else:
            self.arr_breakpoint_A.append(-1)  #store breakpoints
            self.arr_breakpoint_B.append(-1)  #store breakpoints

        ### Characterizing Breakpoint C ### using time gap between consecutive vehicles
        bool_C_found = 0
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
                and bool_C_found == 0 and bool_B_found == 1):

                breakpoint_C = point_of_time[k-1]
                bool_C_found = 1
                self.arr_breakpoint_C.append(breakpoint_C)  #store breakpoints

        if bool_C_found == 0:
            self.arr_breakpoint_C.append(-1)  #store breakpoints

    def C_identification_short_queue(self, start, end, curr_e1_stopbar):
        #Calculating the time gap between vehicles
        #I am assuming that there is maximum only one vehicle per second on the detector
        point_of_time = []
        time_gap_vehicles = []
        time_cnt = 0

        for t in range(start, end+self.phase_length):
            if curr_e1_stopbar['nVehContrib'][t] == 1: #new vehicle enters the detector: start timer new timer and save old measurements
                time_gap_vehicles.append(time_cnt)
                point_of_time.append(t)
                time_cnt=0 #reset time counter for new timing

            if curr_e1_stopbar['nVehContrib'][t] == 0: #timer is running, following vehicle hasn't come yet
                time_cnt= time_cnt+1

        ### Characterizing Breakpoint C ### using time gap between consecutive vehicles
        bool_C_found = 0
        breakpoint_C = 0
        start_search = end-self.duration_green_light + 2 #start searching for C after the green start + 2 seconds and until end
        end_search = end

    ###ATTENTION!! Breakpoint k-1 chosen!!! (alternative k, but that is overestimating!)
        for k in range(0, len(point_of_time)-1):
            if point_of_time[k] >= start_search and point_of_time[k] <= end_search and time_gap_vehicles[k] >= 4 and time_gap_vehicles[k] >= time_gap_vehicles[k-1]:

                breakpoint_C = point_of_time[k-1]
                bool_C_found = 1
                self.arr_breakpoint_C_stopbar.append(breakpoint_C)  #store breakpoints

        if bool_C_found == 0:
            self.arr_breakpoint_C_stopbar.append(-1)  #store breakpoints


    def queue_estimate(self, num_phase, start, end, curr_e1_adv_detector, curr_e1_stopbar):

        self.arr_phase_start.append(start)
        self.arr_phase_end.append(end)        
        #check if breakpoint A exists
        if self.arr_breakpoint_A[len(self.arr_breakpoint_A)-1] == -1:


            #simple input-output method
            if num_phase == 1:
                old_estimated_queue_nVeh = 0
            else:
                old_estimated_queue_nVeh = self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1]*self.k_j

            time_gap = int(self.L_d/self.parent.parent.sumo_network.net.getLane(self.sumolib_in_lane.getID()).getSpeed())
            estimated_queue_nVeh = max(old_estimated_queue_nVeh - self.max_veh_leaving_on_green + sum(curr_e1_adv_detector["nVehContrib"][start-self.duration_green_light:start-time_gap]), 0)+ sum(curr_e1_adv_detector["nVehContrib"][start-time_gap:end-self.duration_green_light])
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
                if num_phase == 1:
                    self.arr_estimated_max_queue_length.append(self.L_d)
                else:
                    self.arr_estimated_max_queue_length.append(self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1])
                self.arr_estimated_time_max_queue.append(end-self.duration_green_light)
                self.used_method.append(2)
            else:
                breakpoint_C = self.arr_breakpoint_C[len(self.arr_breakpoint_C)-1]
                self.v_2 = self.L_d/(breakpoint_B-(end-self.duration_green_light))

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
        return self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1], self.arr_estimated_time_max_queue[len(self.arr_estimated_time_max_queue)-1]

    def get_queue_estimate(self):
        # return  whole estimated queue until now
        return self.arr_estimated_max_queue_length, self.arr_estimated_time_max_queue

    def get_ground_truth_queue(self, num_phase, start, end, curr_e2_detector):
        #calculate ground truth data for queue and store in self arrays
        self.arr_real_max_queue_length.append(sum(curr_e2_detector['startedHalts'][start:end])/self.k_j)
        return sum(curr_e2_detector['startedHalts'][start:end])/self.k_j

    def get_MAPE(self):

        #calculating MAPE for liu + IO
        sum_mape_IO = 0
        cnt = 0
        if len(self.arr_estimated_max_queue_length) > 2:
            for estimation_queue, real_queue in zip(self.arr_estimated_max_queue_length, self.arr_real_max_queue_length):
                if estimation_queue != 0 and real_queue != 0:
                    sum_mape_IO = sum_mape_IO + abs((real_queue-estimation_queue)/real_queue)
                    cnt += 1

            if cnt > 0:
                sum_mape_IO = sum_mape_IO/cnt

            #calculating MAPE for pure liu
            sum_mape_liu = 0
            cnt = 0

            for estimation_queue, real_queue in zip(self.arr_estimated_max_queue_length_pure_liu, self.arr_real_max_queue_length):
                if estimation_queue != 0 and real_queue != 0:
                    sum_mape_liu = sum_mape_liu + abs((real_queue-estimation_queue)/real_queue)
                    cnt += 1
            if cnt > 0:
                sum_mape_liu = sum_mape_liu/cnt

            # in some cases the MAPE is 0 because the estimation is perfect
            # we have to count this cases into the MAPE; check if lane is used
            if sum(self.arr_real_max_queue_length)>0:
                used = True
            else:
                used = False

            return sum_mape_IO, sum_mape_liu, used
        else:
            return -1, -1

    def plot(self, show_graph, show_infos):
        if show_graph == True:
            start = 0
            fig = plt.figure()
            fig.set_figheight(5)
            fig.set_figwidth(5)

            estimation, = plt.plot(self.arr_estimated_time_max_queue, self.arr_estimated_max_queue_length, c='r', label= 'hybrid model')
            ground_truth, = plt.plot(self.arr_estimated_time_max_queue, self.arr_real_max_queue_length, c='b', label= 'ground-truth')
            estimation_pure_liu, = plt.plot(self.arr_estimated_time_max_queue, self.arr_estimated_max_queue_length_pure_liu, c='m', label= 'basic model', linestyle='--')

            plt.legend(handles=[estimation, ground_truth, estimation_pure_liu], fontsize = 18)

            plt.xticks(np.arange(0, 6000, 250))
            plt.xticks(fontsize=18)
            plt.yticks(np.arange(0, 550, 50))
            plt.yticks(fontsize=18)
            plt.xlim(420,1400)
            plt.ylim(0, 200)
            if self.sumolib_in_lane.getID()== '1/0to0/0_0':
                plt.ylim(0, 350)
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

            lane_id = self.sumolib_in_lane.getID()
            dash_lane_id = lane_id.replace('/', '-')
            #just to save plots, delete after writing
            if self.sumolib_in_lane.getID()== '2/2to2/1_2' or self.sumolib_in_lane.getID()== 'top1to1/2_2' or self.sumolib_in_lane.getID()== '1/0to0/0_0':
                fig.savefig("grid_network_"+dash_lane_id+".pdf", bbox_inches='tight')

        if show_infos == True:
            #show some stats for debug
            print('lane id:', self.sumolib_in_lane.getID())
            print('out lane id:', self.sumolib_out_lane.getID())
            print('Estimated queue length: ', self.arr_estimated_max_queue_length)
            print('real queue length: ', self.arr_real_max_queue_length)
            print('phase length:', self.phase_length)
            print('phase start:', self.phase_start)
            print('-----------')

    def get_phase_length(self):
        return self.phase_length

    def get_estimation_data(self, while_running):
        if while_running == True:
            return(self.arr_estimated_time_max_queue[len(self.arr_estimated_time_max_queue)-1],
                   self.arr_real_max_queue_length[len(self.arr_real_max_queue_length)-1], 
                   self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1], 
                   self.arr_estimated_max_queue_length_pure_liu[len(self.arr_estimated_max_queue_length_pure_liu)-1],
                   self.arr_phase_start[len(self.arr_phase_start)-1],
                   self.arr_phase_end[len(self.arr_phase_end)-1])
            
        else:
            return (self.arr_estimated_time_max_queue, self.arr_real_max_queue_length,
                    self.arr_estimated_max_queue_length,
                    self.arr_estimated_max_queue_length_pure_liu,
                    self.arr_phase_start, 
                    self.arr_phase_end)
        
    def get_lane_ID(self):
        return self.sumolib_in_lane.getID()

    def unload_data(self):
        self.parsed_xml_e1_stopbar_detector = None
        self.parsed_xml_e1_adv_detector = None
        self.parsed_xml_e2_detector = None
        self.arr_breakpoint_A = None
        self.arr_breakpoint_B = None
        self.arr_breakpoint_C = None
        self.arr_breakpoint_C_stopbar = None
        self.arr_estimated_max_queue_length = None
        self.arr_estimated_max_queue_length_pure_liu = None
        self.arr_estimated_time_max_queue = None
        self.arr_real_max_queue_length = None
        self.used_method = None
