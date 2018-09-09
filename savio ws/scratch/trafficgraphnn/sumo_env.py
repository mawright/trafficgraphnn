from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register, make
import numpy as np
import time
import six
from collections import namedtuple

import traci
import traci.constants as tc

from trafficgraphnn.sumo_network import SumoNetwork
from trafficgraphnn.trafficlights import get_num_phases


detector_vars = [
    tc.LAST_STEP_OCCUPANCY,
    tc.LAST_STEP_MEAN_SPEED]

timestep_key = tc.VAR_TIME_STEP

LoopPosition = namedtuple('LoopPosition', ['id', 'position'])


class SumoEnv(Env):

    def __init__(
        self, netfile, routefile, additional_flies, policy_type,
        lanewise=True, undirected_graph=False,
        temp_route_file='temp.routes.xml', render=False, seed=10,
        actuation_timestep_secs=30,
        max_timestep=3600
    ):
        super(SumoEnv, self).__init__()
        if render:
            self.binfile = 'sumo-gui'
            raise NotImplementedError
        else:
            self.binfile = 'sumo'

        self.actuation_timestep = actuation_timestep
        self.max_timestep = max_timestep
        self.running = False

        self.lanewise = lanewise
        self.undirected_graph = undirected_graph

        self._seed(seed)

        self.load_sumo_network(netfile, routefile, additional_flies)

    def load_sumo_network(self, netfile, routefile, additional_flies):
        self.sumo_network = self.get_sumo_network(
            netfile, routefile, additional_flies)
        self.trafficlight_ids = [
            tl.getID() for tl in self.sumo_network.net.getTrafficLights()]
        self.graph = self.sumo_network.get_graph()
        self.nodes = self.graph.nodes
        # self.loops_for_node = {}
        # for node, nodedata in self.graph.nodes.items():
        #     loops_for_node[node] = sorted(
        #         [k for k, v in
        #          nodedata.items() if v['type'] == 'e1Detector'],
        #         key=lambda x: nodedata[x]['pos'])

        # max_loops_per_node = max([len(v) for v in loops_for_node.values()])

        self.start()

        self.sim_dt = traci.simulation.getDeltaT()

        # V1: Only control 4x4 intersections with standard turn movements
        self.lights_to_control = [
            light for light in self.trafficlight_ids
            if get_num_phases(light) == 6
        ]

        self.lights_per_num_phases = {}
        for light in traci.trafficlights.getIDList():
            num_phases = get_num_phases(light)
            if num_phases in self.lights_per_num_phases:
                self.lights_per_num_phases[num_phases].append(light)
            else:
                self.lights_per_num_phases[num_phases] = [light]

        self.lights_to_lanes = {
            light: traci.trafficlights.getControlledLanes(light)
            for light in self.lights_to_control
        }

        self.previous_delays_at_light = {
            light: 0.0 for light in traci.trafficlights.getIDList()
        }

        self.lane_to_loops = {
            lane: [] for lane_group in self.lights_to_lanes.values()
            for lane in lane_group
        }
        map(lambda looplist:
            looplist.sort(key=lambda loop: loop.position),
            six.itervalues(self.lane_to_loops))
        self.subscribe_traci_vars()

        self.action_space = spaces.MultiDiscrete([
            [0, self.get_num_phases_for_light(light) - 1]
            for light in self.lights_to_control])

        detector_space = spaces.Box(
            low=float('-inf'), high=float('inf'),
            shape=(len(nodes), max_loops_per_node, len(detector_vars)))

        self.observation_space = detector_space

    def subscribe_traci_vars(self):
        traci.simulation.subscribe(tc.VAR_TIME_STEP)
        for lane_group in six.itervalues(self.lights_to_lanes):
            for lane in lane_group:
                traci.lane.subscribe(lane, [tc.VAR_WAITING_TIME])
                for loop in self.lane_to_loops[loop]:
                    traci.inductionloop.subscribe(
                        loop.id, [detector_vars])

    def start(self):
        if not self.running:
            self.sumo_network.start()
            self.running = True
            self.timestep = 0
            for node_loops in self.loops_for_node.values():
                for loop_id in node_loops:
                    traci.inductionloop.subscribe(loop_id, detector_vars)

    def stop(self):
        if self.running:
            traci.close()
            self.running = False

    def _seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self):
        # reward is net waiting time from previous timestep
        current_timestep_delay = {
            light: sum(
                [traci.lane.getSubscriptionResults(lane)[tc.VAR_WAITING_TIME]
                 for lane in lights_to_lanes[light]])
        }
        local_rewards = {
            light: self.previous_delays_at_light[light]
            - current_timestep_delay[light]
        }
        self.previous_delays_at_light = current_timestep_delay

        global_reward = sum(six.itervalues(local_rewards))

        return local_rewards, global_reward

    def _step(self, action):
        for act, light in zip(action, self.lights_to_control):
            traci.trafficlights.setPhase(light, act)
            traci.trafficlights.setPhaseDuration(
                light, self.actuation_timestep)
        self.timestep += self.actuation_timestep_secs * self.sim_dt
        traci.simulationStep(self.timestep)


        observation = self._observation()
        reward = self._reward()
        done = self.timestep > self.max_timestep

    # def traci_multistep(self, num_sim_timesteps):
    #     observations = np.zeros(
    #         (num_sim_timesteps,) + self.observation_space.shape)
    #     rewards =
    #     for t in range(num_sim_timesteps):

    def _reset(self):
        self.stop()
        time.sleep(1)
        self.start()
        observation = self._observation()
        return observation

    def _render(self, mode='human', close=False):
        pass

    def get_sumo_network(self, netfile, routefile, addlfiles, sumo_seed=None):
        if sumo_seed is None:
            sumo_seed = np.random.randint(1e6)
        return SumoNetwork(
            netfile, lanewise=self.lanewise,
            undirected_graph=self.undirected_graph, routefile=routefile,
            addlfiles=addlfiles, binfile=self.binfile, seed=sumo_seed)

    def build_policy(self, sumo_network):


# register(
    # id='Traffic-Single-Grid-v0',
    # entry_point='trafficgraphdl.envs:Traffic-Grid',
# )
