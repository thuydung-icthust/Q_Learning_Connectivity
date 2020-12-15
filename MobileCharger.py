from scipy.spatial import distance

import Parameter as para
from MobileCharger_Method import get_location, charging


class MobileCharger:
    def __init__(self, energy=None, e_move=None, start=para.depot, end=para.depot, velocity=None,
                 e_self_charge=None, capacity=None):
        self.is_stand = False  # is true if mc stand and charge
        self.is_self_charge = False  # is true if mc is charged
        self.is_active = False  # is false if none of node request and mc is standing at depot

        self.start = start  # from location
        self.end = end  # to location
        self.current = start  # location now
        self.end_time = -1  # the time when mc finish charging

        self.energy = energy  # energy now
        self.capacity = capacity  # capacity of mc
        self.e_move = e_move  # energy for moving
        self.e_self_charge = e_self_charge  # energy receive per second
        self.velocity = velocity  # velocity of mc

        self.list_request = []  # the list of request node

    def update_location(self, func=get_location):
        self.current = func(self)
        self.energy -= self.e_move

    def charge(self, network=None, node=None, charging_func=charging):
        charging_func(self, network, node)

    def self_charge(self):
        self.energy = min(self.energy + self.e_self_charge, self.capacity)

    def check_state(self):
        if distance.euclidean(self.current, self.end) < 1:
            self.is_stand = True
            self.current = self.end
        else:
            self.is_stand = False
        if distance.euclidean(para.depot, self.end) < 10 ** -3:
            self.is_self_charge = True
        else:
            self.is_self_charge = False

    def get_next_location(self, network, time_stem, index_f, optimizer=None):
        next_location, charging_time = optimizer.update(network)
        f_name = "log/DiscreteQ/MC_avtivity_" + str(index_f) + ".txt"
        data = str([self.energy, self.current, next_location, charging_time])
        with open(f_name, "a+") as f:
            f.write(data)
            f.write("\n")
        # print("next state =", self.action_list[self.state], self.state, charging_time)
        self.start = self.current
        next_p = (int(next_location[0]), int(next_location[1]))
        self.end = next_p
        moving_time = distance.euclidean(self.start, self.end) / self.velocity
        self.end_time = time_stem + moving_time + charging_time

    def run(self, network, time_stem, index_f, optimizer=None):
        # print(self.energy, self.start, self.end, self.current)
        if (not self.is_active and self.list_request) or abs(
                time_stem - self.end_time) < 1:
            self.is_active = True
            self.list_request = [request for request in self.list_request if
                                 network.node[request["id"]].energy < network.node[request["id"]].energy_thresh]
            if not self.list_request:
                self.is_active = False
            self.get_next_location(network=network, time_stem=time_stem, optimizer=optimizer, index_f = index_f)
        else:
            if self.is_active:
                if not self.is_stand:
                    print("moving")
                    self.update_location()
                elif not self.is_self_charge:
                    print("charging")
                    self.charge(network)
                else:
                    print("self charging")
                    self.self_charge()
        if self.energy < para.E_mc_thresh and not self.is_self_charge and self.end != para.depot:
            self.start = self.current
            self.end = para.depot
            self.is_stand = False
            charging_time = self.capacity / self.e_self_charge
            moving_time = distance.euclidean(self.start, self.end) / self.velocity
            self.end_time = time_stem + moving_time + charging_time
        self.check_state()
