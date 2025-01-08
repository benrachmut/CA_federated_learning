from enum import Enum
from random import random

from matplotlib import pyplot as plt


class NetType(Enum):
    ALEXNET = "AlexNet"
    VGG = "VGG"

class DataSet(Enum):
    CIFAR100 = "CIFAR100"
    CIFAR10 = "CIFAR10"



class ExpType(Enum):
    IID_diff_nets = 1
    NonIID_diff_nets = 2
    IID_same_nets = 3
    NonIID_same_nets = 4
    short = 5


class ExperimentConfig:
    def __init__(self):
        self.num_classes = None
        self.identical_clients = None
        self.mix_percentage = None
        self.server_net_type = None
        self.client_net_type = None
        self.batch_size = None
        self.learning_rate_train_s = None

        self.data_set_selected = DataSet.CIFAR10
        self.seed_num = 1
        self.with_weight_memory = True
        self.with_server_net = True
        self.epochs_num_input = 2#20
        self.epochs_num_train = 2#10

        self.iterations = 12
        self.server_split_ratio = 0.2
        self.learning_rate = 0.001
        self.learning_rate_train_c = 0.001
        self.num_clusters = 1

        # ----------------

        # ----------------

        self.num_clients = None  # num_classes*identical_clients
        self.percent_train_data_use = 0.05
        self.percent_test_relative_to_train = 0.05

    def update_type_of_experiment(self,exp_type):
        flag = False
        if exp_type == ExpType.IID_diff_nets:
            self.num_classes = 10
            self.identical_clients = 1
            self.mix_percentage = 1
            self.server_net_type = NetType.VGG
            self.client_net_type = NetType.ALEXNET
            self.batch_size = 64
            self.learning_rate_train_s = 0.0001
            flag = True
        elif exp_type == ExpType.NonIID_diff_nets:
            self.num_classes = 10
            self.identical_clients = 1
            self.mix_percentage = 0.2
            self.server_net_type = NetType.VGG
            self.client_net_type = NetType.ALEXNET
            self.batch_size = 128
            self.learning_rate_train_s = 0.0001
            flag = True

        elif exp_type == ExpType.IID_same_nets:
            self.num_classes = 10
            self.identical_clients = 1
            self.mix_percentage = 1
            self.server_net_type = NetType.ALEXNET
            self.client_net_type = NetType.ALEXNET
            self.batch_size = 64
            self.learning_rate_train_s = 0.001
            self.num_clients = self.num_classes * self.identical_clients
            flag = True

        elif exp_type == ExpType.NonIID_same_nets:
            self.num_classes = 10
            self.identical_clients = 1
            self.mix_percentage = 0.2
            self.server_net_type = NetType.ALEXNET
            self.client_net_type = NetType.ALEXNET
            self.batch_size = 128
            self.learning_rate_train_s = 0.001
            flag = True

        if flag:
            self.num_clients = self.num_classes * self.identical_clients

        if exp_type == ExpType.short:
            self.num_classes = 3
            self.identical_clients = 2
            self.mix_percentage = 0.2
            self.server_net_type = NetType.ALEXNET
            self.client_net_type = NetType.ALEXNET
            self.batch_size = 128
            self.learning_rate_train_s = 0.001
            self.num_clients = self.num_classes * self.identical_clients


experiment_config = ExperimentConfig()






