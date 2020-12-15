import csv

from scipy.spatial import distance

import Parameter as para
from Network_Method import uniform_com_func, to_string, count_package_function


class Network:
    def __init__(self, list_node=None, mc=None, target=None):
        self.node = list_node  # list of node in network
        self.set_neighbor()  # find neighbor of each node
        self.set_level()  # set the distance in graph from each node to base
        self.mc = mc  # mobile charger
        self.target = target  # list of target. each item is index of sensor where target is located

    def set_neighbor(self):
        for node in self.node:
            for other in self.node:
                if other.id != node.id and distance.euclidean(node.location, other.location) <= node.com_ran:
                    node.neighbor.append(other.id)

    def set_level(self):
        queue = []
        for node in self.node:
            if distance.euclidean(node.location, para.base) < node.com_ran:
                node.level = 1
                queue.append(node.id)
        while queue:
            for neighbor_id in self.node[queue[0]].neighbor:
                if not self.node[neighbor_id].level:
                    self.node[neighbor_id].level = self.node[queue[0]].level + 1
                    queue.append(neighbor_id)
            queue.pop(0)

    def communicate(self, communicate_func=uniform_com_func):
        return communicate_func(self)

    def run_per_second(self, t, optimizer, index_f):
        state = self.communicate()
        request_id = []
        for index, node in enumerate(self.node):
            if node.energy < node.energy_thresh and node.is_active:
                node.request(mc=self.mc, t=t)
                request_id.append(index)
            else:
                node.is_request = False
        if request_id:
            for index, node in enumerate(self.node):
                if index not in request_id and (t - node.check_point[-1]["time"]) > 50:
                    node.set_check_point(t)
        if optimizer:
            self.mc.run(network=self, time_stem=t, optimizer=optimizer, index_f=index_f)
        return state

    def simulate_lifetime(self, optimizer, index, file_name="log/energy_log.csv"):
        filename = "log/Lifetime/qlearning_output" + str(index)
        filename_txt = "log/Lifetime/qlearning_output_" + str(index) +".txt"
        energy_log = open(filename, "a+")
        writer = csv.DictWriter(energy_log, fieldnames=["time", "mc energy", "min energy"])
        writer.writeheader()
        t = 0
        while self.node[self.find_min_node()].energy >= 0:
            t = t + 1
            if t % 100 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
                current_package = self.count_package()
                if current_package != nb_package:
                    nb_package = current_package
                data = str([t, nb_package, self.mc.energy, self.mc.current,
                            self.node[self.find_min_node()].energy]) + "\n"
                with open(filename_txt, "a+") as f:
                    f.write(data)
            state = self.run_per_second(t, optimizer, index)
            if not (t - 1) % 50:
                writer.writerow(
                    {"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})

        writer.writerow({"time": t, "mc energy": self.mc.energy, "min energy": self.node[self.find_min_node()].energy})
        energy_log.close()

    def simulate_max_time(self, optimizer, index, max_time=10000, file_name="log/information_log.csv"):
        file_name = "log/DiscreteQ/information_log" + str(index) +".csv"
        filenametxt = "log/DiscreteQ/networkinfor" + str(index) +".txt"
        # f = open(file_name, "r+")
        # f.truncate(0)
        # f.close()
        # f = open(filenametxt, "r+")
        # f.truncate(0)
        # f.close()
        information_log = open(file_name, "a+")
        writer = csv.DictWriter(information_log, fieldnames=["time", "nb dead", "nb package"])
        writer.writeheader()
        nb_dead = self.count_dead_node()
        nb_package = len(self.target)
        t = 0

        while t <= max_time:
            t += 1
            for node in self.node:
                node.check_active(self)
            if t % 100 == 0:
                print(t, self.mc.current, self.node[self.find_min_node()].energy)
                data = str([t, nb_dead, nb_package, self.mc.energy, self.mc.current, self.node[self.find_min_node()].energy]) + "\n"
                with open(filenametxt, "a+") as f:
                    f.write(data)

                writer.writerow({"time": t, "nb dead": nb_dead, "nb package": nb_package})
                # writer.writerow(data)
            state = self.run_per_second(t, optimizer, index)
            current_dead = self.count_dead_node()
            current_package = self.count_package()
            if current_dead != nb_dead or current_package != nb_package:
                nb_dead = current_dead
                nb_package = current_package

        information_log.close()

    def simulate(self, optimizer, index,  max_time=None, file_name="log/energy_log.csv"):
        if max_time:
            self.simulate_max_time(optimizer=optimizer, index=index, max_time=max_time, file_name=file_name)
        else:
            self.simulate_lifetime(optimizer=optimizer, index = index, file_name=file_name)

    def print_net(self, func=to_string):
        func(self)

    def find_min_node(self):
        min_energy = 10 ** 10
        min_id = -1
        for node in self.node:
            if not node.is_active:
                continue
            if node.energy < min_energy:
                min_energy = node.energy
                min_id = node.id
        return min_id

    def count_dead_node(self):
        count = 0
        for node in self.node:
            if node.energy < node.min_energy:
                count += 1
        return count

    def count_package(self, count_func=count_package_function):
        count = count_func(self)
        return count
