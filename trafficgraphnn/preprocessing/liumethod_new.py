import multiprocessing
from collections import Iterable
from itertools import repeat

import numpy as np
import pandas as pd

from trafficgraphnn.preprocessing.io import (green_times_from_lane_light_df,
                                             queueing_intervals_from_lane_light_df,
                                             sumo_output_xmls_to_hdf,
                                             tls_output_xml_to_hdf)
from trafficgraphnn.utils import DetInfo

JAM_DENSITY = 0.13333 # hardcoded default jam density value (veh/meter)


# codes for estimation methods
__INPUT_OUTPUT = 'input-output'
__SATURATED = 'saturated'
__LIU = 'liu'


def liu_method_for_net(sumo_network, output_data_hdf_filename,
                       jam_density=JAM_DENSITY, num_workers=None,
                       use_lane_change_accounting_heuristic=True):
    args = []
    lane_ids = []
    idds = []
    for lane_id, lane_data in sumo_network.graph.nodes.data():
        if 'detectors' not in lane_data:
            continue
        idds.append(get_length_between_loop_detectors(sumo_network, lane_id))
        lane_ids.append(lane_id)

    args = zip(repeat(output_data_hdf_filename), lane_ids, idds,
               repeat(jam_density))
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.starmap(liu_for_lane, args)

    out = {result.columns[0]: result for result in results}
    if use_lane_change_accounting_heuristic:
        out = postprocess_negative_from_lane_changes(sumo_network, out)

    return out


def postprocess_negative_from_lane_changes(sn, results, num_workers=None):
    """Heuristic to try and mitigate negative net loop flows from lane changes.

    Lane queue estimates computed by the simple input output method will
    miss lane-changing vehicles. This function identifies the lanes that
    have an apparent negative queue from input-output accounting and
    decreases the road's other lanes queues by that amount as a heuristic.
    """
    grouped_lanes = _split_lanes_by_edges(sn, results)
    dfs = {}
    with multiprocessing.Pool(num_workers) as pool:
        edge_dfs = pool.map(distribute_io_estimate_deficit_for_edge,
                            grouped_lanes.values())

    dfs.update((lane, lane_df.dropna())
                for edge_df in edge_dfs
                for lane, lane_df in edge_df.groupby(axis=1, level=0))

    return dfs


def distribute_io_estimate_deficit_for_edge(lane_estimate_dfs_for_edge):
    lane_ids = [df.columns[0] for df in lane_estimate_dfs_for_edge]
    df = pd.concat(lane_estimate_dfs_for_edge,
                   axis=1, join='outer', keys=lane_ids)
    df.columns = pd.MultiIndex.from_product([lane_ids,
                                                ['estimate', 'method']])

    is_io_estimate = df.loc[:, (slice(None), 'method')] == __INPUT_OUTPUT
    io_estimates = (df.loc[:, (slice(None), 'estimate')]
                    * is_io_estimate.values)

    deficit_assigned = distribute_deficit_loop(io_estimates)
    df.update(deficit_assigned)
    df.loc[:, (slice(None), 'estimate')
            ] = df.loc[:, (slice(None), 'estimate')].clip(lower=0)

    return df


def distribute_deficit_loop(df):
    while df[df < 0].count().sum() > 0:
        df = distribute_deficit_one_iter(df)
    return df


def distribute_deficit_one_iter(df):
    negatives = df[df < 0]
    positives = df[df > 0]
    sum_negs = negatives.sum(axis=1)
    num_pos = positives.count(axis=1)

    to_add = sum_negs / num_pos
    added = positives.add(to_add, axis=0)

    return added


def _split_lanes_by_edges(sn, results):
    lane_groups = {}
    lanes = list(results.keys())
    while len(lanes) > 0:
        lane = lanes[0]
        edge = sn.net.getLane(lane).getEdge()
        all_lanes_in_edge = [ln.getID() for ln in edge.getLanes()]
        lane_group = []
        for ln in all_lanes_in_edge:
            try:
                lane_group.append(results.pop(ln))
            except KeyError:
                continue
            lanes.remove(ln)
        lane_groups[edge.getID()] = lane_group

    return lane_groups


def raw_output_data_to_hdf(sumo_network):
    output_dir = sumo_network.detector_data_path
    output_hdf = sumo_output_xmls_to_hdf(output_dir)

    light_switch_out_files = set()
    for edge in sumo_network.graph.edges.data():
        try:
            light_switch_out_files.add(edge[-1]['tls_output_info']['dest'])
        except KeyError:
            continue

    assert len(light_switch_out_files) == 1 # more than one xml not supported yet

    tls_output_xml_to_hdf(light_switch_out_files.pop())

    return output_hdf


def get_length_between_loop_detectors(sumo_network, lane_id):
    det_info = sumo_network.graph.nodes[lane_id]['detectors']
    e1_dets = [DetInfo(k, v) for k, v in det_info.items()
               if v['type'] == 'e1Detector']

    e1_dets.sort(key=lambda x: x.info['pos'])

    stopbar_pos = float(e1_dets[-1].info['pos'])
    advance_pos = float(e1_dets[-2].info['pos'])

    return abs(stopbar_pos - advance_pos)


def liu_for_lane(output_data_hdf_filename, lane_id, inter_detector_distance,
                 jam_density=JAM_DENSITY):
    stopbar_detector_id = 'e1_' + lane_id + '_0'
    advance_detector_id = 'e1_' + lane_id + '_1'

    with pd.HDFStore(output_data_hdf_filename, 'r') as store:

        stopbar_detector_df = store['raw_xml/' + stopbar_detector_id].copy()
        advance_detector_df = store['raw_xml/' + advance_detector_id].copy()

        lane_light_series = store['raw_xml/tls_switch'][lane_id].copy()
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

    breakpoint_A_list, breakpoint_B_list = breakpoints_A_B_wholedf(
        advance_detector_df, queueing_periods)

    # breakpoint C: only look up until the next queue starts forming (i.e.,
    # the start of the next queueing period)
    breakpoint_C_list = [breakpoint_C(advance_detector_df, B, next_intvl[0])
                         for B, next_intvl in zip(breakpoint_B_list,
                                                  queueing_periods[1:])]

    estimates_list = []
    residual_queue_estimate = 0
    for stop, adv, green, A, B, C in zip(stopbar_queueing_periods,
                                         advance_queueing_periods,
                                         green_times,
                                         breakpoint_A_list,
                                         breakpoint_B_list,
                                         breakpoint_C_list):
        estimate = queue_estimate(stop, adv, green, A, B, C,
                                 residual_queue_estimate,
                                 inter_detector_distance, jam_density)
        if isinstance(estimate[0], Iterable): # can return one or two tuples
            estimates_list.extend(estimate)
        else:
            estimates_list.append(estimate)

        residual_queue_estimate = max(estimates_list[-1][1], 0)

    # return pandas df of the time that a liu estimate is made
    # and the estimate itself
    result = pd.DataFrame(
        estimates_list, columns=['begin',
                                 '{}'.format(lane_id),
                                 '{}_method'.format(lane_id)])
    result = result.set_index('begin')
    return result
    # return estimates_list


def breakpoints_A_B(advance_df):
    binary_occupancy = (advance_df['occupancy'] >= 100).astype(np.bool)

    # breakpoint A: the first time where the detector has a static car on it
    # (we see if it is fully occupied for 4 seconds)
    bkpt_A = breakpoint_A_for_period(binary_occupancy)

    # breakpoint B: the first second after breakpoint A that the detector is
    # not occupied
    bkpt_B = breakpoint_B_for_period(binary_occupancy, bkpt_A)

    if bkpt_A is None or bkpt_B is None:
        return None, None

    return bkpt_A, bkpt_B


def breakpoints_A_B_wholedf(advance_df, queueing_intervals):
    binary_occupancy = (advance_df['occupancy'] >= 100).astype(bool)

    breakpoints_A = [breakpoint_A_for_period(binary_occupancy.loc[q[0]:q[1]])
                     for q in queueing_intervals]

    breakpoints_B = [breakpoint_B_for_period(binary_occupancy.loc[q[0]:q[1]],
                                             A)
                     for q, A in zip(queueing_intervals, breakpoints_A)]

    return breakpoints_A, breakpoints_B


def breakpoint_A_for_period(binary_occupancy):
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


def breakpoint_B_for_period(binary_occupancy, bkpt_A):
    """Find breakpoint B.

    Breakpoint B is the first timestep after breakpoint B that the detector
    is not continuously occupied. This means it is the timestep that the
    discharge wave has reached the advance detector"""
    if bkpt_A is None:
        return None
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
                   breakpoint_A, breakpoint_B, breakpoint_C,
                   prev_phase_queue_estimate, inter_detector_distance,
                   jam_density):
    if breakpoint_A is None:
        time, estimate = input_output_method(stopbar_df, advance_df,
                                             prev_phase_queue_estimate,
                                             green_start)
        return time, estimate, __INPUT_OUTPUT
    elif breakpoint_C is None:
        # No breakpoint C means that the queue is saturated and therefore
        # unobservable. In this case the Liu method gives no guidance, so we
        # choose to estimate that the queue reaches the advance detector
        phase_end = stopbar_df.index[-1]
        saturated_queue_estimate_veh = inter_detector_distance * jam_density
        return phase_end, saturated_queue_estimate_veh, __SATURATED
    else:
        estimate_or_estimates = liu_estimate_from_breakpoints(
            advance_df, stopbar_df, green_start, breakpoint_B, breakpoint_C,
            inter_detector_distance, jam_density)
        if isinstance(estimate_or_estimates[0], Iterable):
            return tuple(est + (__LIU,) for est in estimate_or_estimates)
        return estimate_or_estimates + (__LIU,)

def input_output_method(stopbar_df, advance_df, prev_phase_queue_estimate,
                        green_time):
    total_outflow = stopbar_df.loc[:,'nVehContrib'].sum()
    total_inflow = advance_df.loc[:,'nVehContrib'].sum()
    net_flow = total_inflow - total_outflow

    queue_estimate = prev_phase_queue_estimate + net_flow
    assert stopbar_df.index[-1] == advance_df.index[-1]
    t_queue_estimate = stopbar_df.index[-1]
    return t_queue_estimate, queue_estimate


def liu_estimate_from_breakpoints(advance_df, stopbar_df, green_start,
                                  breakpoint_B,
                                  breakpoint_C, inter_detector_distance,
                                  jam_density):
    """Liu's "Expansion I" method (Equation 11)"""

    # estimate shockwave speed

    if breakpoint_B != green_start:
        v2 = inter_detector_distance / (breakpoint_B - green_start)
        v2 = abs(v2)
    else:
        v2 = np.inf

    # max queue
    n = advance_df.loc[:breakpoint_C, 'nVehContrib'].sum()
    max_queue_veh = n + inter_detector_distance * jam_density # units: veh

    # time of max queue
    max_queue_m = max_queue_veh / jam_density
    t_queue_max = np.round(green_start + max_queue_m / v2)

    outputs = (t_queue_max, max_queue_veh)

    # queue at end of green phase (if max queue not at end of green phase)
    # estimated by subtracting outflow after time of max queue
    # and adding inflow after breakpoint C (pre-breakpoint C arrivals included
    # in max queue)
    t_phase_end = advance_df.index[-1]
    if t_phase_end > t_queue_max:
        post_max_queue_outflow = stopbar_df.loc[t_queue_max:,
                                                'nVehContrib'].sum()
        post_breakpoint_C_inflow = advance_df.loc[breakpoint_C:,
                                                  'nVehContrib'].sum()

        phase_end_queue_veh = (max_queue_veh - post_max_queue_outflow
                            + post_breakpoint_C_inflow)

        outputs = outputs, (t_phase_end, phase_end_queue_veh)

    # speed of the queue discharge wave
    # v3 = max_queue_m - inter_detector_distance / (breakpoint_C - t_queue_max)
    # assert v3 > 0
    # Liu et al. draw a distinction between the queue at the end of the green
    # light and the minimum queue; we will simply place the minimum queue at
    # end of the green light
    # t_min_queue = phase_end_t + min_queue_m / v4

    return outputs


def _split_df_by_intervals(df, intervals):
    return [df.loc[(interval[0] <= df.index) & (df.index <= interval[1])]
            for interval in intervals]
