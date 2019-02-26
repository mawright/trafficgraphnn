import numpy as np

from trafficgraphnn.preprocessing.io import (green_times_from_lane_light_df,
                                             light_timing_xml_to_phase_df,
                                             queueing_intervals_from_lane_light_df,
                                             sumo_output_xmls_to_hdf)
from trafficgraphnn.utils import DetInfo

JAM_DENSITY = 0.13333 # hardcoded default jam density value (veh/meter)


def liu_method(sumo_network, output_data_hdf_filename, light_timing_df):
    pass


def load_output_data(sumo_network):
    output_dir = sumo_network.detector_data_path
    output_hdf = sumo_output_xmls_to_hdf(output_dir)

    light_switch_out_files = set()
    for edge in sumo_network.graph.edges.data():
        try:
            light_switch_out_files.add(edge[-1]['tls_output_info']['dest'])
        except KeyError:
            continue

    assert len(light_switch_out_files) == 1 # more than one xml not supported yet

    green_df = light_timing_xml_to_phase_df(light_switch_out_files.pop())

    return output_hdf, green_df


def get_lane_length(sumo_network, lane_id):
    return sumo_network.net.getLane(lane_id).getLength()


def get_length_between_loop_detectors(sumo_network, lane_id):
    det_info = sumo_network.graph.nodes[lane_id]['detectors']
    e1_dets = [DetInfo(k, v) for k, v in det_info.items()
               if v['type'] == 'e1Detector']

    e1_dets.sort(key=lambda x: x.info['pos'])

    stopbar_pos = float(e1_dets[-1].info['pos'])
    advance_pos = float(e1_dets[-2].info['pos'])

    return abs(stopbar_pos - advance_pos)


def liu_for_lane(store, lane_id, inter_detector_distance, lane_length,
                 jam_density=None):
    if jam_density is None:
        jam_density = JAM_DENSITY
    stopbar_detector_id = 'e1_' + lane_id + '_0'
    advance_detector_id = 'e1_' + lane_id + '_1'

    stopbar_detector_df = store['raw_xml/' + stopbar_detector_id]
    advance_detector_df = store['raw_xml/' + advance_detector_id]

    lane_light_series = store['raw_xml/tls_switch'][lane_id]
    queueing_periods = queueing_intervals_from_lane_light_df(lane_light_series)
    queueing_periods = list(queueing_periods)
    green_times = green_times_from_lane_light_df(lane_light_series)
    green_times = list(green_times)

    # if the first green time is before the first queueing period (i.e. light
    # starts green), chop it off
    if green_times[0] < queueing_periods[0][0]:
        green_times = green_times[1:]

    # break up the df's by queueing periods
    stopbar_queueing_periods = _split_df_by_intervals(stopbar_detector_df,
                                                      queueing_periods)
    advance_queueing_periods = _split_df_by_intervals(advance_detector_df,
                                                      queueing_periods)

    breakpoint_A_list, breakpoint_B_list = zip(*[breakpoints_A_B(q) for q
                                                 in advance_queueing_periods])

    # breakpoint C: only look up until the next queue starts forming (i.e.,
    # the start of the next queueing period)
    breakpoint_C_list = [breakpoint_C(advance_detector_df, B, next_intvl[0])
                         for B, next_intvl in zip(breakpoint_B_list,
                                                  queueing_periods[1:])]

    liu_estimate_list = []
    estimate = 0
    for stop, adv, green, A, B, C in zip(stopbar_queueing_periods,
                                         advance_queueing_periods,
                                         green_times,
                                         breakpoint_A_list,
                                         breakpoint_B_list,
                                         breakpoint_C_list):
        prev_estimate = estimate
        estimate = queue_estimate(stop, adv, green, A, B, C, prev_estimate,
                                  inter_detector_distance, jam_density,
                                  lane_length)
        liu_estimate_list.append(estimate)

    queueing_interval_ends = [q[1] for q in queueing_periods]

    # return tuples of the time that a liu estimate is made (the end of a
    # queueing/discharging period) and the estimate itself
    return zip(queueing_interval_ends, liu_estimate_list)


def breakpoints_A_B(advance_df):
    binary_occupancy = (advance_df['occupancy'] >= 100).astype(np.bool)

    # breakpoint A: the first time where the detector has a static car on it
    # (we see if it is fully occupied for 4 seconds)
    bkpt_A = breakpoint_A(binary_occupancy)

    # breakpoint B: the first second after breakpoint A that the detector is
    # not occupied
    bkpt_B = breakpoint_B(binary_occupancy, bkpt_A)

    if bkpt_A is None or bkpt_B is None:
        return None, None

    return bkpt_A, bkpt_B


def breakpoint_A(binary_occupancy):
    """Find breakpoint A.

    Breakpoint A is the first timestep where the detector has a static car on
    it that stays there until the queue discharges (we see if the detector is
    continuously occupied for 4 seconds)"""
    continuously_occupied = (binary_occupancy[::-1]
                             .rolling(4)
                             .agg(lambda x: x.all())[::-1]
                             .fillna(0))
    bkpt = continuously_occupied[continuously_occupied == 1]
    try:
        bkpt = bkpt.index[0]
    except IndexError: # no breakpoints, the detector is not continuously occupied
        return None
    return bkpt


def breakpoint_B(binary_occupancy, bkpt_A):
    """Find breakpoint B.

    Breakpoint B is the first timestep after breakpoint B that the detector
    is not continuously occupied. This means it is the timestep that the
    discharge wave has reached the advance detector"""
    continuously_occupied = (binary_occupancy.loc[bkpt_A:]
                                             .expanding()
                                             .agg(lambda x: x.all()))
    bkpt = continuously_occupied.loc[continuously_occupied == 0]
    try:
        bkpt = bkpt.index[0]
    except IndexError:
        return None
    return bkpt


def breakpoint_C(detector_df, breakpoint_B, end_search_time, min_time_gap=3):
    """Find Breakpoint C.

    Breakpoint C is after breakpoint B, and is the timestep where there is a
    time gap between discharging vehicles, indicating that the expansion wave
    has reached the back of the built-up queue"""
    if breakpoint_B is None: # no breakpoint B means no breakpoint C
        return None
    pattern = np.array([*([0] * min_time_gap), 1])
    matched = (detector_df.loc[breakpoint_B:end_search_time, 'nVehContrib']
                          .rolling(len(pattern))
                          .apply(lambda x: np.equal(pattern, x).all(),
                                 raw=False) # suppress a pandas warning
                          .fillna(0))
    try:
        breakpoint_C = matched[matched == 1].index[0]
    except IndexError: # no breakpoint C
        return None
    return breakpoint_C


def queue_estimate(stopbar_df, advance_df, green_start,
                   prev_phase_queue_estimate, breakpoint_A, breakpoint_B,
                   breakpoint_C, inter_detector_distance, jam_density,
                   lane_length):
    if breakpoint_A is None:
        return input_output_method(stopbar_df, advance_df,
                                   prev_phase_queue_estimate, green_start)
    elif breakpoint_C is None:
        # No breakpoint C means that the queue is saturated and therefore
        # unobservable. In this case the Liu method gives no guidance, so we
        # choose to estimate that the lane is fully backed up.
        return lane_length * jam_density
    else:
        return liu_estimate_from_breakpoints(advance_df, green_start,
                                             breakpoint_B, breakpoint_C,
                                             inter_detector_distance,
                                             jam_density)


def input_output_method(stopbar_df, advance_df, prev_phase_queue_estimate,
                        green_time):
    total_outflow = stopbar_df.loc[green_time:,'nVehContrib'].sum()
    total_inflow = advance_df.loc[:,'nVehContrib'].sum()
    net_flow = total_inflow - total_outflow

    queue_estimate = prev_phase_queue_estimate + net_flow
    assert queue_estimate > 0
    return max(queue_estimate, 0)


def liu_estimate_from_breakpoints(advance_df, green_start, breakpoint_B,
                                  breakpoint_C, inter_detector_distance,
                                  jam_density):
    """Liu's "Expansion I" method"""

    # estimate shockwave speed (does not seem to be needed unless we care about
    # the time of max queue)
    # v2 = inter_detector_distance / (breakpoint_B - green_start)
    # v2 = -abs(v2)

    # number of vehicles that pass the advance detector between start of
    # queueing interval and breakpoint C
    n = advance_df.loc[:breakpoint_C, 'nVehContrib'].sum()
    queue_estimate = n + inter_detector_distance * jam_density
    return queue_estimate


def _split_df_by_intervals(df, intervals):
    return [df.loc[(interval[0] <= df.index) & (df.index <= interval[1])]
            for interval in intervals]
