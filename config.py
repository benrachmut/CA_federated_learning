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

exp_type = ExpType.IID_diff_nets

if exp_type == ExpType.IID_diff_nets:
    num_classes = 20
    identical_clients = 1
    mix_percentage = 1
    server_net_type = NetType.VGG
    client_net_type = NetType.ALEXNET

if exp_type == ExpType.NonIID_diff_nets:
    num_classes = 10
    identical_clients = 2
    mix_percentage = 0.2
    server_net_type = NetType.VGG
    client_net_type = NetType.ALEXNET

if exp_type == ExpType.IID_same_nets:
    num_classes = 20
    identical_clients = 1
    mix_percentage = 1
    server_net_type = NetType.ALEXNET
    client_net_type = NetType.ALEXNET

if exp_type == ExpType.NonIID_same_nets:
    num_classes = 10
    identical_clients = 2
    mix_percentage = 0.2
    server_net_type = NetType.ALEXNET
    client_net_type = NetType.ALEXNET


data_set_selected = DataSet.CIFAR10
seed_num = 1

with_server_net = True
epochs_num_input = 10
iterations = 50
server_split_ratio = 0.2
batch_size = 32
learning_rate = 0.001
learning_rate_train = 0.0001
#----------------



num_clients = num_classes*identical_clients
#----------------
num_clusters = 1
percent_train_data_use = 0.1
percent_test_relative_to_train = 1



summary = (
            f"num_clusters_{num_clusters}_"
            f"Mix_Percentage_{mix_percentage}_"
            f"Epochs_{epochs_num_input}_"
            f"Iterations_{iterations}_"
            f"Server_Split_Ratio_{server_split_ratio}_"
            f"Num_Classes_{num_classes}_"
            f"Identical_Clients_{identical_clients}"
            f"with_server_net{with_server_net}"
            f"server_net{server_net_type.name}"
            f"client_net{client_net_type.name}")

#epochs_num_input_train_client = 10
#server_epochs_num_input = 10


#client_batch_size_train = 32
#client_learning_rate_train = 0.001

#client_batch_size_fine_tune = 32
#client_learning_rate_fine_tune = 0.001

#client_batch_size_evaluate = 32

#server_batch_size_train = 32
#server_learning_rate_train = 0.0001

#server_batch_size_evaluate = 32


#def get_CIFAR10_superclass_dict():
#    dict_ = {"animal":['bird', 'cat', 'deer', 'dog', 'frog', 'horse'],
#             "vehicle":['airplane', 'automobile','ship', 'truck']
#             }
#    return dict_

def get_meta_data():
    ans = {
        'c_amount':[num_clients],
        'seed':[seed_num],
        'server_data': [server_split_ratio],
        'is_server_net': [with_server_net],  # You might need to pass or save client_split_ratio
        'epochs': [epochs_num_input],
        'percent_train_data': [percent_train_data_use]
    }
    return ans

def get_meta_data_text_keys():
    ans = []
    for k in get_meta_data().keys():
        ans.append(k)
    return ans

def file_name():
    ans = ""
    for k,v in get_meta_data().items():
        ans = ans+k+"_"+str(v[0])+"__"
    return ans

