from __future__ import absolute_import, print_function, division

import os
import logging

import sumolib

from trafficgraphnn.utils import iterfy

logger = logging.getLogger(__name__)


def adjust_detector_length(requested_detector_length,
                           requested_distance_to_tls,
                           lane_length):
    """ Adjusts requested detector's length according to
        the lane length and requested distance to TLS.

        If requested detector length is negative, the resulting detector length
        will match the distance between requested distance to TLS and lane
        beginning.


        If the requested detector length is positive, it will be adjusted
        according to the end of lane ending with TLS: the resulting length
        will be either the requested detector length or, if it's too long
        to be placed in requested distance from TLS, it will be shortened to
        match the distance between requested distance to TLS
        and lane beginning. """

    if requested_detector_length == -1:
        return lane_length - requested_distance_to_tls

    return min(lane_length - requested_distance_to_tls,
               requested_detector_length)


def adjust_detector_position(final_detector_length,
                             requested_distance_to_tls,
                             lane_length):
    """ Adjusts the detector's position. If the detector's length
        and the requested distance to TLS together are longer than
        the lane itself, the position will be 0; it will be
        the maximum distance from lane end otherwise (taking detector's length
        and requested distance to TLS into accout). """

    return max(0,
               lane_length - final_detector_length - requested_distance_to_tls)


detector_type_tag_dict = {
    'e1': 'e1Detector',
    'e2': 'e2Detector'
}


def generate_detector_set(netfile, detector_type, distance_to_tls,
                          detector_def_file=None,
                          detector_output_file=None, detector_length=None,
                          frequency=60):

    if detector_type not in ['e1', 'e2']:
        raise ValueError('Unknown detector type: {}'.format(detector_type))

    if detector_length is not None and not (
        len(distance_to_tls) == len(detector_length)
        or len(distance_to_tls) == 1
        or len(detector_length) == 1
    ):
        raise ValueError((
            'Input size mismatch: len(distance_to_tls) = {} '
            'and len(detector_length) = {}').format(
            len(distance_to_tls), len(detector_length)))

    default_net_config_dir = os.path.dirname(os.path.realpath(netfile))
    net_name = os.path.basename(
        os.path.splitext(os.path.splitext(netfile)[0])[0])
    default_output_data_dir = os.path.join(default_net_config_dir, 'output')
    if detector_def_file is None:
        detector_def_file = os.path.join(
            default_net_config_dir,
            '{}_{}.add.xml'.format(
                net_name, detector_type)
        )
    if detector_output_file is None:
        detector_output_file = os.path.join(
            default_output_data_dir,
            '{}_{}_output.xml'.format(
                net_name, detector_type)
        )

    detector_tag = detector_type_tag_dict[detector_type]
    relative_output_filename = os.path.relpath(
        detector_output_file,
        os.path.dirname(detector_def_file))

    detectors_xml = sumolib.xml.create_document("additional")
    lanes_with_detectors = set()

    net = sumolib.net.readNet(netfile)

    for tls in net._tlss:
        for connection in tls._connections:
            lane = connection[0]
            lane_length = lane.getLength()
            lane_id = lane.getID()

            logger.debug("Creating detector for lane %s" % (str(lane_id)))

            if lane_id in lanes_with_detectors:
                logger.warn(
                    "Detectors for lane %s already generated" % (str(lane_id)))
                continue

            lanes_with_detectors.add(lane_id)

            for i, distance in enumerate(distance_to_tls):
                detector_xml = detectors_xml.addChild(detector_tag)
                detector_xml.setAttribute("file", relative_output_filename)
                detector_xml.setAttribute("freq", str(frequency))
                detector_xml.setAttribute("friendlyPos", "x")
                detector_xml.setAttribute(
                    "id",
                    "{}_{}_{}".format(
                        detector_type, lane_id, i).replace('/', '-')
                )
                detector_xml.setAttribute("lane", str(lane_id))

                if detector_type == 'e2':
                    final_detector_length = adjust_detector_length(
                        detector_length[i],
                        distance,
                        lane_length)
                    detector_xml.setAttribute(
                        "length", str(final_detector_length))
                else:
                    final_detector_length = 0

                final_detector_position = adjust_detector_position(
                    final_detector_length,
                    distance,
                    lane_length)
                detector_xml.setAttribute("pos", str(final_detector_position))

    detector_file = open(os.path.realpath(detector_def_file), 'w')
    detector_file.write(detectors_xml.toXML())
    detector_file.close()

    if not os.path.exists(os.path.dirname(detector_output_file)):
        os.makedirs(os.path.dirname(detector_output_file))

    return os.path.realpath(detector_def_file)


def generate_e1_detectors(netfile, distance_to_tls, detector_def_file=None,
                          detector_output_file=None, frequency=60):
    return generate_detector_set(
        netfile, 'e1', iterfy(distance_to_tls), detector_def_file,
        detector_output_file, frequency=frequency)


def generate_e2_detectors(netfile, distance_to_tls, detector_def_file=None,
                          detector_output_file=None, detector_length=250,
                          frequency=60):

    distance_to_tls = iterfy(distance_to_tls)
    detector_length = iterfy(detector_length)

    if len(detector_length) not in [1, len(distance_to_tls)]:
        raise ValueError(
            'detector_length must be 1 or len(distance_to_tls), got {}'.format(
                len(detector_length))
        )

    if len(detector_length) == 1:
        detector_length = [detector_length] * len(distance_to_tls)

    return generate_detector_set(
        netfile, 'e2', distance_to_tls, detector_def_file,
        detector_output_file, detector_length, frequency)
