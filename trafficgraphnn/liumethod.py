import os, sys
import time
 #inserted from sumo wiki
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:   
    sys.exit("please declare environment variable 'SUMO_HOME'")

import logging
from trafficgraphnn.utils import iterfy, get_net_dir, get_net_name

import sumolib.net
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
#import traci._lane

_logger = logging.getLogger(__name__)


class LiuEtAlRunner(object):
    def __init__(self, sumo_network, lane_subset=None, time_window=None):
        self.sumo_network = sumo_network
        #self.sumo_network.load_data_to_graph()
        self.graph = self.sumo_network.get_graph()
        self.net = sumo_network.net
        self.parsed_xml_e1_detectors = None
        self.bool_parsed_e1 = 0
        self.parsed_xml_e2_detectors = None
        self.bool_parsed_e2 = 0
        self.parsed_xml_tls = None
        self.bool_parsed_tls = 0
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
            for intersection in self.liu_intersections:
                intersection.run_next_phase(num_phase)                   
        pass

    def run_next_phase(self, num_phase):
        # run the single-phase method for each intersection
        for intersection in self.liu_intersections:
            intersection.run_next_phase(num_phase)
        pass
    
    def plot_results_every_lane(self):
        for intersection in self.liu_intersections:
            intersection.plot_results()
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
            out_lane_id = out_lane.getID()
            if in_lane_id not in liu_lanes_id:
                #if in_lane_id == '1/3to1/2_0' or in_lane_id == '1/3to1/2_1' or in_lane_id == '1/3to1/2_2': #selecting the eligible lanes (remove later on!)
                
                liu_lanes_id.append(in_lane_id)
                print('Creating Liu lane with in_lane = ', in_lane_id, 'and out_lane = ', out_lane_id)
                lane = LiuLane(in_lane, out_lane, self)
                self.liu_lanes.append(lane)                     
                
                            
    def run_next_phase(self, num_phase):
         #run the single-phase calculation for each lane
         for current_lane in self.liu_lanes:
             start, end, curr_e1_stopbar, curr_e1_adv_detector, curr_e2_detector = current_lane.parse_cycle_data(num_phase)
             current_lane.breakpoint_identification(num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector)
             current_lane.queue_estimate(num_phase, start, end, curr_e1_adv_detector)
             current_lane.get_ground_truth_queue(num_phase, start, end, curr_e2_detector)
             mape = current_lane.get_MAPE()
             print('MAPE: ', mape)        
         pass
     
    def plot_results(self):
        for lane in self.liu_lanes:
            lane.plot()
        pass
        

class LiuLane(object):
    def __init__(self, sumolib_in_lane, out_lane, parent):
        # initialize this object: lane id, references to proper objects in the
        # big sumo_network file, and objects to store results
        # as well as tracithe state: last step/time period processed
        # (and leftover queue for if we need to use the input/output method in
        # any steps)
        self.sumolib_in_lane = sumolib_in_lane
        self.sumolib_out_lane = out_lane
        self.max_veh_leaving_on_green = 15
        
        #dataframes that are loaded once at the beginning
        self.df_e1_adv_detector = pd.DataFrame()
        self.df_e1_stopbar = pd.DataFrame()
        self.df_e2_detector = pd.DataFrame()
        self.df_traffic_lights = pd.DataFrame()
        self.graph = parent.parent.graph
        
        columns_e1_adv = ['time', 'occupancy', 'nVehEntered', 'nVehContrib'] #later just use this one the df above are not necessary
        columns_e1_stopbar = ['time', 'nVehContrib']
        self.curr_e1_adv_detector = pd.DataFrame(columns = columns_e1_adv)
        self.curr_e1_stopbar = pd.DataFrame(columns = columns_e1_stopbar)
        columns_e2 = ['time', 'startedHalts']
        self.curr_e2_detector = pd.DataFrame(columns = columns_e2)        

#        
#        self.df_e1_adv_detector = self.graph.node[sumolib_lane]['detectors']['e1_'+lane+'_1']['data_series']
#        self.df_e1_stopbar = self.graph.node[sumolib_lane]['detectors']['e1_'+lane+'_0']['data_series']
#        self.df_e2_detector = self.graph.node[sumolib_lane]['detectors']['e2_'+lane+'_0']['data_series']                  
#        self.df_traffic_lights = self.graph.edges[(sumolib_in_lane.getID(), out_lane.getID())]['switch_times']
#        self.phase_length = int(self.df_traffic_lights['begin'].values[1] - self.df_traffic_lights['begin'].values[0])
#        self.duration_green_light = int(self.df_traffic_lights['duration'].values[0])
#        self.max_num_phases = int(6000/(self.phase_length))  ###Attention: right simu time!
#        print("All dataframes for lane " + lane +" are loaded")
        
        
        #arrays for store results
        self.arr_breakpoint_A = []
        self.arr_breakpoint_B = []
        self.arr_breakpoint_C = []
        
        self.arr_estimated_max_queue_length = []
        self.arr_estimated_time_max_queue = []
        self.arr_real_max_queue_length = []
        self.used_method = []
        
        #parameters for lane        
        lane_id = self.sumolib_in_lane.getID()
        dash_lane_id = lane_id.replace('/', '-')
        adv_detector_id = "e1_" + dash_lane_id + "_1"
        stopbar_detector_id = "e1_" + dash_lane_id + "_0"
        
        out_lane_id = self.sumolib_out_lane.getID()

        self.L_d = float(self.graph.nodes('detectors')[self.sumolib_in_lane.getID()][stopbar_detector_id]['pos']
            )-float(self.graph.nodes('detectors')[self.sumolib_in_lane.getID()][adv_detector_id]['pos'])
        self.k_j = 0.13333 #jam density
        
        self.parent = parent
        #self.time_window = time_window

        #estimating tls data!
        if self.parent.parent.parsed_xml_tls == None:
            self.parent.parent.parsed_xml_tls = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                        self.graph.edges[lane_id, out_lane_id]['tls_output_info']['dest']))
            print('tls data parsing was successful')
            self.parent.parent.bool_parsed_tls = 1
         
        cnt = 0  
        got_phase_length = 0
        got_duration = 0
        for node in self.parent.parent.parsed_xml_tls.getroot():   
            fromLane = str(node.attrib.get('fromLane'))
            if fromLane == lane_id and (got_phase_length == 0 or got_duration == 0):
                if cnt == 0:
                   self.phase_start = float(node.attrib.get('end'))
                   self.duration_green_light = float(node.attrib.get('duration'))
                   got_duration = 1
                   cnt = 1
                elif cnt == 1:
                   phase_end = float(node.attrib.get('end'))
                   self.phase_length = int(phase_end - self.phase_start)
                   got_phase_length = 1
                   
        self.parent.parent.parsed_xml_tls = None       
            

        # we would like to be able to have each lane's calculation of its Liu
        # state be entirely local to its instance of this class

    def run_next_phase(self):
        # run one iteration of the liu method for the next phase in the
        # data history
        pass

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
       
        if self.parent.parent.parsed_xml_e1_detectors == None:
        #if self.parent.parent.bool_parsed_e1 == 0:
            self.parent.parent.parsed_xml_e1_detectors = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                                 self.graph.nodes('detectors')[lane_id][adv_detector_id]['file']))
            print('e1 detector parsing was successful')
            self.parent.parent.bool_parsed_e1 = 1
        
        list_time =[]       #for advanced detector
        list_occupancy =[]
        list_nVehEntered =[]
        list_nVehContrib =[]
        
        list_time_stop = []     #for stop bar detector
        list_nVehContrib_stop = []
        
        for node in self.parent.parent.parsed_xml_e1_detectors.getroot():
            begin = float(node.attrib.get('begin'))
            det_id = node.attrib.get('id')
            if begin >= start-self.phase_length and begin < end+self.phase_length and det_id == adv_detector_id:
                list_time.append(begin)
                list_occupancy.append(float(node.attrib.get('occupancy')))
                list_nVehEntered.append(float(node.attrib.get('nVehEntered')))
                list_nVehContrib.append(float(node.attrib.get('nVehContrib')))
                
            if begin >= start-self.phase_length and begin < end+self.phase_length and det_id == stopbar_detector_id:
                list_time_stop.append(begin)
                list_nVehContrib_stop.append(float(node.attrib.get('nVehContrib')))
                
        self.curr_e1_adv_detector = pd.DataFrame(
                {'time': list_time, 'occupancy': list_occupancy,
                 'nVehEntered': list_nVehEntered, 'nVehContrib': list_nVehContrib})
        self.curr_e1_stopbar = pd.DataFrame(
                {'time': list_time, 'nVehContrib': list_nVehContrib})
                          
        self.curr_e1_adv_detector.set_index('time', inplace=True)  
        self.curr_e1_stopbar.set_index('time', inplace=True) 

        #self.parent.parent.parsed_xml_e1_detectors = None

        ### parse e2 detector from xml file ###

        e2_detector_id = "e2_" + dash_lane_id + "_0"
        
        
        if self.parent.parent.parsed_xml_e2_detectors == None:
            self.parent.parent.parsed_xml_e2_detectors = et.parse(
                    os.path.join(os.path.dirname(self.parent.parent.sumo_network.netfile),
                                 self.graph.nodes('detectors')[lane_id][e2_detector_id]['file'])) 
            print('e2 detector parsing was successful')
            self.parent.parent.bool_parsed_e2 = 1
        
        list_time_e2 = []
        list_startedHalts_e2 = []
        
        for node in self.parent.parent.parsed_xml_e2_detectors.getroot():
            begin = float(node.attrib.get('begin'))
            det_id = node.attrib.get('id')
            if begin >= start and begin < end+self.phase_length and det_id == e2_detector_id:
                list_time_e2.append(begin)
                list_startedHalts_e2.append(float(node.attrib.get('startedHalts')))
                                
        self.curr_e2_detector = pd.DataFrame(
                {'time': list_time_e2, 'startedHalts': list_startedHalts_e2})     
        self.curr_e2_detector.set_index('time', inplace=True)  
    
        #self.parent.parent.parsed_xml_e2_detectors = None
        
        return start, end, self.curr_e1_stopbar, self.curr_e1_adv_detector, self.curr_e2_detector

    def breakpoint_identification(self, num_phase, start, end, curr_e1_stopbar, curr_e1_adv_detector):
                                
        #Calculate binary occupancy
        binary_occ_t = pd.Series()
        binary_occ_t = curr_e1_adv_detector['occupancy'].apply(lambda x:(1 if x >= 100 else 0))

    
        #Calculating the time gap between vehicles
        #I am assuming that there is maximum only one vehicle per second on the detector
        point_of_time = []
        time_gap_vehicles = []
        time_cnt=0
    
        for t in range(start, end+self.phase_length):
            if curr_e1_adv_detector['nVehEntered'][t] == 1: #new vehicle enters the detector: start timer new timer and save old measurements
                time_gap_vehicles.append(time_cnt)
                point_of_time.append(t)
                time_cnt=0 #reset time counter for new timing
    
            if curr_e1_adv_detector['nVehEntered'][t] == 0: #timer is running, following vehicle hasn't come yet
                time_cnt= time_cnt+1    
                                
    
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
            #print("breakpoints in phase", num_phase)
            #print(breakpoint_A)
            #print(breakpoint_B)
            self.arr_breakpoint_A.append(breakpoint_A)  #store breakpoints
            self.arr_breakpoint_B.append(breakpoint_B)  #store breakpoints
            
            #estimating how many vehicles can leave the lane during green phase! (for each lane)
            max_veh_leaving = sum(curr_e1_stopbar["nVehContrib"][end-self.duration_green_light:end])           
            if max_veh_leaving > self.max_veh_leaving_on_green:
                self.max_veh_leaving_on_green = max_veh_leaving
                #print('Found higher max_veh_leaving on green =', self.max_veh_leaving_on_green)
            
        else:
            #print("No reliable breakpoints in phase", num_phase, "found!")
            self.arr_breakpoint_A.append(-1)  #store breakpoints
            self.arr_breakpoint_B.append(-1)  #store breakpoints        

        ### Characterizing Breakpoint C ### using time gap between consecutive vehicles
        bool_C_found = 0
        breakpoint_C = 0
        start_search = breakpoint_B + 10 #start searching for C after the breakpoint B + 10 seconds and until end; little offset of 10 sec is necessary to avoid influence from breakpoint B
        end_search = end + 50
                
    ###ATTENTION!! Breakpoint k-1 chosen!!! (alternative k, but that is overestimating!)
        for k in range(0, len(point_of_time)-1):
            if point_of_time[k] >= start_search and point_of_time[k] <= end_search and time_gap_vehicles[k] >= 4 and time_gap_vehicles[k] >= time_gap_vehicles[k-1] and point_of_time[k-1] >= breakpoint_B and bool_C_found ==0 and bool_B_found ==1:
                #print("Breakpoint C found!")
                #print("Breakpoint:", point_of_time[k-1], "Time Gap:", time_gap_vehicles[k-1])
                                
                breakpoint_C = point_of_time[k-1]
                bool_C_found = 1
                self.arr_breakpoint_C.append(breakpoint_C)  #store breakpoints

        if bool_C_found == 0:
           # print("No breakpoint C in phase", num_phase, "found!") 
            self.arr_breakpoint_C.append(-1)  #store breakpoints

       
        # next iteration of breakpoint-finding...
        pass


    def queue_estimate(self, num_phase, start, end, curr_e1_adv_detector):
              
        #check if breakpoint A exists
        if self.arr_breakpoint_A[len(self.arr_breakpoint_A)-1] == -1: #no breakpoint A exists -> short queue estimation method
            
            #simple input-output method
            #print('simple input-output method in phase', num_phase)
            if num_phase == 1:
                old_estimated_queue_nVeh = 0
            else:
                old_estimated_queue_nVeh = self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1]*self.k_j
            
            estimated_queue_nVeh = max(old_estimated_queue_nVeh - self.max_veh_leaving_on_green + sum(curr_e1_adv_detector["nVehContrib"][start-self.duration_green_light:start-8]), 0)+ sum(curr_e1_adv_detector["nVehContrib"][start-8:end-self.duration_green_light])
            self.arr_estimated_max_queue_length.append(estimated_queue_nVeh/self.k_j)
            #estimated_queue_nVeh = max(old_estimated_queue_nVeh - 17 + sum(df_e1_adv_detector["nVehContrib"][start-duration_green_light-8:start-8]), 0)+ sum(df_e1_adv_detector["nVehContrib"][start-8:end-duration_green_light])
            self.arr_estimated_time_max_queue.append(end-self.duration_green_light)
            
            self.used_method.append(0)
            
        else:
            #breakpoint_A = self.arr_breakpoint_A[len(self.arr_breakpoint_A)-1]
            breakpoint_B = self.arr_breakpoint_B[len(self.arr_breakpoint_B)-1]

            if self.arr_breakpoint_C[len(self.arr_breakpoint_C)-1] == -1 or self.arr_breakpoint_C[len(self.arr_breakpoint_A)-2] == self.arr_breakpoint_A[len(self.arr_breakpoint_A)-1]:
                #print('OVERSATURATION, no Breakpoint C found!!!')
                if num_phase == 1:
                    self.arr_estimated_max_queue_length.append(self.L_d)
                else:
                    self.arr_estimated_max_queue_length.append(self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1])
                self.arr_estimated_time_max_queue.append(end-self.duration_green_light)
                self.used_method.append(2)
            else:
                breakpoint_C = self.arr_breakpoint_C[len(self.arr_breakpoint_C)-1]
                v_2 = self.L_d/(breakpoint_B-(end-self.duration_green_light))
                #print("Estimated v_2: ", v_2)
                
                ### Expansion I      
                #print('Liu method extension I in phase', num_phase)
                #n is the number of vehicles passing detector between T_ng(start red phase) and T_C (breakpoint C)
                n = sum(curr_e1_adv_detector["nVehEntered"].loc[(end-self.duration_green_light):breakpoint_C])
                L_max = n/self.k_j + self.L_d
                T_max = (end-self.duration_green_light) + L_max/abs(v_2)
                
                self.arr_estimated_max_queue_length.append(L_max)
                self.arr_estimated_time_max_queue.append(T_max)
                self.used_method.append(1)                
                
        pass
        # next iteration of queue estimation from breakpoints

    def get_last_queue_estimate(self):
        # return last-estimated queue
        return self.arr_estimated_max_queue_length[len(self.arr_estimated_max_queue_length)-1], self.arr_estimated_time_max_queue[len(self.arr_estimated_time_max_queue)-1]

    def get_queue_estimate(self):
        # return  whole estimated queue until now
        return self.arr_estimated_max_queue_length, self.arr_estimated_time_max_queue

    def get_ground_truth_queue(self, num_phase, start, end, curr_e2_detector):
        #calculate ground truth data for queue and store in self arrays
        #print('ppppppp', sum(curr_e2_detector['startedHalts'][start:end]))
        self.arr_real_max_queue_length.append(sum(curr_e2_detector['startedHalts'][start:end])/self.k_j)
        return sum(curr_e2_detector['startedHalts'][start:end])/self.k_j
       
    def get_MAPE(self):
        #calculate MAPE
        sum_mape = 0
        #print('-----', self.arr_real_max_queue_length) #debug
        if len(self.arr_estimated_max_queue_length) > 2:
            for estimation_queue, real_queue in zip(self.arr_estimated_max_queue_length, self.arr_real_max_queue_length):
                if estimation_queue != 0 and real_queue != 0:
                    sum_mape = sum_mape + abs((real_queue-estimation_queue)/real_queue)
            
            sum_mape = sum_mape/len(self.arr_estimated_max_queue_length)
            return sum_mape
        else:
            #print('Too less values to calculate MAPE!!!')
            return -1

    def plot(self):
        # can add the plotting code here...
        start = 0
        fig = plt.figure()
        fig.set_figheight(8)
        fig.set_figwidth(16)
        
        estimation, = plt.plot(self.arr_estimated_time_max_queue, self.arr_estimated_max_queue_length, c='r', label= 'estimation')
        ground_truth, = plt.plot(self.arr_estimated_time_max_queue, self.arr_real_max_queue_length, c='b', label= 'ground-truth')
        plt.legend(handles=[estimation, ground_truth], fontsize = 18)
        plt.ylim(0, 300)
        

        
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
        
        #show some stats for debug
        print('lane id:', self.sumolib_in_lane.getID())
        print('out lane id:', self.sumolib_out_lane.getID())
        print('Estimated queue length: ', self.arr_estimated_max_queue_length)
        print('real queue length: ', self.arr_real_max_queue_length)
        print('phase length:', self.phase_length)
        print('phase start:', self.phase_start)
        
        
        pass
