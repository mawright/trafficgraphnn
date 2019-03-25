"""
Code for visualizations.
"""
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from trafficgraphnn.load_data import pad_value_for_feature
from trafficgraphnn.preprocessing.io import \
    green_phase_start_ends_from_lane_light_df


def get_figure_dir(results_filename):
    results_dir = os.path.dirname(results_filename)
    return os.path.join(results_dir, 'figures')


def plot_results_for_file(filename):
    fig_dir = get_figure_dir(filename)
    with pd.HDFStore(filename, 'r') as store:
        prefixes = prefixes_in_store(store)

        for prefix in prefixes:
            os.makedirs(os.path.join(fig_dir, 'queue_estimate', prefix),
                        exist_ok=True)
            os.makedirs(os.path.join(fig_dir, 'vehseen_estimate', prefix),
                        exist_ok=True)
            lanes = store[prefix + '/X'].index.get_level_values('lane').unique()

            for lane in lanes:
                _plot_for_lane(filename ,fig_dir, prefix, lane)


def prefixes_in_store(store):
    keys = store.keys()
    prefixes = [re.search('(?<=/).+(?=/X|/Y)', key).group() for key in keys]
    return sorted(list(set(prefixes)))


def _plot_for_lane(store_filename, output_dir, prefix, lane_id):
    with pd.HDFStore(store_filename, 'r') as store:
        liu_series = (store[prefix + '/X'].loc[:, 'liu_estimated_veh']
                                          .xs(lane_id, level='lane'))
        max_jam_series = (store[prefix + '/Y']
                          .loc[:, 'e2_0/maxJamLengthInVehicles']
                          .xs(lane_id, level='lane'))
        predicted_max_jamseries = (store[prefix + '/Yhat']
                                  .loc[:, 'e2_0/maxJamLengthInVehicles']
                                  .xs(lane_id, level='lane'))

        green_series = (store[prefix + '/X'].loc[:, 'green']
                                            .xs(lane_id, level='lane'))
        vehseen_series = (store[prefix + '/Y'].loc[:, 'e2_0/nVehSeen']
                                              .xs(lane_id, level='lane'))
        predicted_vehseen_series = (store[prefix + '/Yhat']
                                    .loc[:, 'e2_0/nVehSeen']
                                    .xs(lane_id, level='lane'))

    fig, ax = lane_queue_liu_vs_nn(liu_series, max_jam_series,
                                    predicted_max_jamseries)
    fig.savefig(os.path.join(output_dir, 'queue_estimate', prefix,
                             '{}.eps'.format(lane_id)),
                bbox_inches='tight')
    plt.close(fig)

    fig, ax = lane_nvehseen_plot(green_series, vehseen_series,
                                 predicted_vehseen_series)
    fig.savefig(os.path.join(output_dir, 'vehseen_estimate', prefix,
                             '{}.eps'.format(lane_id)),
                bbox_inches='tight')
    plt.close(fig)


def lane_queue_liu_vs_nn(liu_series, max_jam_series, predicted_series):
    non_pad_timesteps = (max_jam_series
                         != pad_value_for_feature['maxJamLengthInVehicles'])

    fig, ax = plt.subplots()
    ax.plot(max_jam_series.loc[non_pad_timesteps]
                          .reset_index(drop=True, name='cycle'),
                          'k', label='True cycle queue')

    ax.plot(liu_series.loc[non_pad_timesteps]
                      .reset_index(drop=True, name='cycle'),
            'b--', label='Physics-based estimate')
    ax.plot(predicted_series.loc[non_pad_timesteps]
                            .reset_index(drop=True, name='cycle'),
            'r:', label='Neural-net estimate')
    ax.legend()
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Queue (vehicles)')
    return fig, ax


def lane_nvehseen_plot(true_series, predicted_series, green_series=None):
    fig, ax = plt.subplots()

    # color green and red lights
    if green_series is not None:
        try:
            green_phases = green_phase_start_ends_from_lane_light_df(green_series)
            ax.axvspan(green_series.index[0], green_series.index[-1],
                    alpha=0.5, color='red') # red background
            for phase in green_phases:
                ax.axvspan(phase[0], phase[1], alpha=0.5, color='green')
        except IndexError: # the light never changes
            pass

    ax.plot(true_series, label='True')
    ax.plot(predicted_series, label='Prediction')
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vehicles on lane')
    return fig, ax
