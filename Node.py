import math
from scipy.spatial import distance

import Parameter as para
from Node_Method import to_string, find_receiver, request_function, estimate_average_energy


class Node:
    def __init__(self, location=None, com_ran=None, sen_ran=None, energy=None, prob=para.prob, avg_energy=0.0,
                 len_cp=10, id=None, is_active=True, energy_max=None, energy_thresh=None):
        self.location = location  # location of sensor
        self.com_ran = com_ran  # communication range
        self.sen_ran = sen_ran  # sensing range
        self.energy = energy  # energy of sensor
        self.energy_max = energy_max  # capacity of sensor
        self.energy_thresh = energy_thresh # threshold to sensor send request for mc
        self.prob = prob  # probability of sending data
        self.check_point = [{"E_current": self.energy, "time": 0, "avg_e": 0.0}]  # check point of information of sensor
        self.used_energy = 0.0  # energy was used from last check point to now
        self.avg_energy = avg_energy  # average energy of sensor
        self.len_cp = len_cp  # length of check point list
        self.id = id  # identify of sensor
        self.neighbor = []  # neighborhood of sensor
        self.is_active = is_active  # statement of sensor. If sensor dead, state is False
        self.is_request = False  # if node requested, is_request = True
        self.level = 0  # the distance from node to base
        self.min_energy = 0.02

    def set_average_energy(self, func=estimate_average_energy):
        self.avg_energy = func(self)

    def set_check_point(self, t):
        if len(self.check_point) >= self.len_cp:
            self.check_point.pop(0)
        self.check_point.append(
            {"E_current": self.energy, "time": t, "avg_e": self.used_energy / (t - self.check_point[-1]["time"])})
        self.avg_energy = self.check_point[-1]["avg_e"]
        self.used_energy = 0.0

    def charge(self, mc):
        if not self.is_active:
            return 0
        if self.energy <= self.energy_max - 10 ** -5 and mc.is_stand and self.is_active:
            d = distance.euclidean(self.location, mc.current)
            p_theory = para.alpha / (d + para.beta) ** 2
            p_actual = min(self.energy_max - self.energy, p_theory)
            self.energy = self.energy + p_actual
            return p_actual
        else:
            return 0

    def send(self, net=None, package=None, receiver=find_receiver, is_energy_info=False):
        d0 = math.sqrt(para.EFS / para.EMP)
        package.update_path(self.id)
        if not self.is_active:
            return
        if distance.euclidean(self.location, para.base) > self.com_ran:
            receiver_id = receiver(self, net)
            if receiver_id != -1:
                d = distance.euclidean(self.location, net.node[receiver_id].location)
                e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
                self.energy -= e_send * package.size
                self.used_energy += e_send * package.size
                net.node[receiver_id].receive(package)
                net.node[receiver_id].send(net, package, receiver, is_energy_info)
        else:
            package.is_success = True
            d = distance.euclidean(self.location, para.base)
            e_send = para.ET + para.EFS * d ** 2 if d <= d0 else para.ET + para.EMP * d ** 4
            self.energy -= e_send * package.size
            self.used_energy += e_send * package.size
            package.update_path(-1)
        self.check_active(net)

    def receive(self, package):
        self.energy -= para.ER * package.size
        self.used_energy += para.ER * package.size

    def check_active(self, net):
        if self.energy < self.min_energy or len(self.neighbor) == 0:
            self.is_active = False
            self.is_request = False
        else:
            a = [1 for neighbor in self.neighbor if net.node[neighbor].is_active]
            self.is_active = True if len(a) > 0 else False

    def request(self, mc, t, request_func=request_function):
        self.set_check_point(t)
        # print(self.check_point)
        if not self.is_request:
            request_func(self, mc, t)
            self.is_request = True

    def print_node(self, func=to_string):
        func(self)
