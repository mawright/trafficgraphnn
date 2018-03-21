import traci


def logics_equal(logic_x, logic_y):
    return str(logic_x) == str(logic_y)


def phases_equal(logic_x, logic_y):
    return str(logic_x._phases) == str(logic_y._phases)


def get_num_phases(traffic_light):
    return len(
        traci.trafficlights.getCompleteRedYellowGreenDefinition(
            traffic_light)[0]._phases
    )
