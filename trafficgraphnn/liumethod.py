import logging
from utils import iterfy

import sumolib.net

_logger = logging.getLogger(__name__)


class LiuEtAlRunner(object):
    def __init__(self, sumo_network, lane_subset=None, time_window=None):
        self.sumo_network = sumo_network
        self.net = sumo_network.net
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
                    ' network and wil be ignored: {}'.format(
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

    def run_up_to_time(self, time_end):
        # iterate on the single-step methods for each intersection until
        # reaching the given time
        pass

    def run_next_phase(self):
        # run the single-phase method for each intersection
        pass


class LiuIntersection(object):
    # object that holds several LiuLane objects corresponding to its component
    # lanes as well as intersection-level info like traffic light definition
    def __init__(self, sumolib_tls, parent, time_window):
        self.sumolib_tls = sumolib_tls
        self.parent = parent
        self.time_window = time_window

        self.liu_lanes = []
        for conn in self.sumolib_tls.getConnections():
            in_lane, out_lane, conn_id = conn
            # create a LiuLane for each in-lane
            # update this call if the instance lane needs more references to
            # eg dataframes
            lane = LiuLane(in_lane, self)
            self.liu_lanes.append(lane)
            # etc...

    def run_next_phase(self):
        # run the single-phase calculation for each lane
        pass


class LiuLane(object):
    def __init__(self, sumolib_lane, parent, time_window=None):
        # initialize this object: lane id, references to proper objects in the
        # big sumo_network file, and objects to store results
        # as well as the state: last step/time period processed
        # (and leftover queue for if we need to use the input/output method in
        # any steps)
        self.sumolib_lane = sumolib_lane
        self.parent = parent
        self.time_window = time_window

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

    def breakpoint_identification(self):
        pass
        # next iteration of breakpoint-finding...

    def queue_estimate(self):
        pass
        # next iteration of queue estimation from breakpoints

    def get_queue_estimate(self):
        pass
        # return last-estimated queue

    def plot(self):
        # can add the plotting code here...
        pass
