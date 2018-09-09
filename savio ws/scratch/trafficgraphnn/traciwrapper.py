import traci


class TraciWrapper(object):
    def __init__(self, sumo_network):
        self.sumo_network = sumo_network
        self.tls_to_controlled_dict = self.sumo_network.tls_to_controlled_dict

        self.running = False

        self.timestep = 0

        self.loop_ids = None
        self.loop_to_lane = None
        self.loop_to_edge = None
        self.trafficlight_ids = None

    def start(self):
        if not self.running:
            self.sumo_network.start()
            self.running = True

            self.loop_ids = self.get_loop_ids()
            self.loop_to_lane = self.get_loop_to_lane_dict()
            self.loop_to_edge = self.get_loop_to_edge_dict()

            self.trafficlight_ids = self.get_trafficlight_ids()

    def get_trafficlight_ids(self):
        return traci.trafficlights.getIDList()

    def get_loop_ids(self):
        return traci.inductionloop.getIDList()

    def get_loop_to_lane_dict(self):
        loop_list = self.get_loop_ids()
        loop_to_lane = {
            loop_id: traci.inductionloop.getLaneID(loop_id)
            for loop_id in loop_list
        }
        return loop_to_lane

    def get_edge_ids(self):
        return traci.edge.getIDList()

    def get_loop_to_edge_dict(self, loop_to_lane=None):
        loop_list = self.get_loop_ids()
        if loop_to_lane is None:
            loop_to_lane = self.get_loop_to_lane_dict()

        loop_to_edge = {
            loop_id: traci.lane.getEdgeID(
                traci.inductionloop.getLaneID(loop_to_lane[loop_id])
            ) for loop_id in loop_list
        }
        return loop_to_edge
