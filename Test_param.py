from Node import Node
import random
from Network import Network
import pandas as pd
from ast import literal_eval
from MobileCharger import MobileCharger
from Q__Learning import Q_learning
from Inma import Inma
def Test(file_name = "data/thaydoitileguitin.csv", des_log = "log/change_alpha/", size_tt = 2, ord = 0, alpha_p = 0.2, theta_p = 0.2):

    for index in range(1,size_tt):
        df = pd.read_csv(file_name)
        node_pos = list(literal_eval(df.node_pos[index]))
        list_node = []
        for i in range(len(node_pos)):
            location = node_pos[i]
            com_ran = df.commRange[index]
            energy = df.energy[index]
            energy_max = df.energy[index]
            prob = df.freq[index]
            node = Node(location=location, com_ran=com_ran, energy=energy, energy_max=energy_max, id=i,
                    energy_thresh=0.4 * energy, prob=prob)
            list_node.append(node)
        mc = MobileCharger(energy=df.E_mc[index], capacity=df.E_max[index], e_move=df.e_move[index],
                    e_self_charge=df.e_mc[index], velocity=df.velocity[index])
        target = [int(item) for item in df.target[index].split(',')]
        net = Network(list_node=list_node,  mc=mc, target=target)
        print(len(net.node), len(net.target), max(net.target))
        q_learning = Q_learning(network=net, alpha_p = alpha_p, theta_p = theta_p)
        inma = Inma()
        net.simulate(optimizer= q_learning, index= ord, file_name= des_log)

if __name__ == "__main__":
    # filename = ["data/thaydoitileguitin.csv", "data/thaydoisonode.csv", "data/thaydoinangluongmc.csv"]
    # des_log = ["log/Lifetime/Frequency/", "log/Lifetime/NodeNum/", "log/Lifetime/PowerMC/"]
    # size_tt = [4, 6, 5]
    # for i in range(0,3):
    #     Test(file_name= filename[i], des_log=des_log[i], size_tt= size_tt[i])
    alpha_param = [0.1,0.2,0.3,0.4,0.5]
    theta_param = [0.1,0.2,0.3,0.4,0.5]
    for i in range(4,5):
        Test(ord=i, alpha_p = alpha_param[i], theta_p=0.2)