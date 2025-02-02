import pickle

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

import config

from config import *
from functions import *
from entities import *


class RecordData:
    def __init__(self,records_dict):#loss_measures,accuracy_measures,accuracy_pl_measures,accuracy_measures_k,accuracy_pl_measures_k):
        self.records_dict = records_dict
        #self.loss_measures = loss_measures
        #self.accuracy_pl_measures = accuracy_pl_measures
        #self.accuracy_test_measures = accuracy_measures
        #self.accuracy_test_measures_k_half_cluster[t] = accuracy_measures_k
        #self.accuracy_pl_measures_k_half_cluster[t] = accuracy_pl_measures_k

        self.summary = (
            f"clusters_{experiment_config.num_clusters}_"
            f"batch_{experiment_config.batch_size}_"
            f"trLRC_{experiment_config.learning_rate_train_c}_"
            f"trainLR_{experiment_config.learning_rate_train_s}_"
            f"tuneLR_{experiment_config.learning_rate_fine_tune_c}_"
            f"Mix_P_{experiment_config.mix_percentage}_"
            f"Epoch_{experiment_config.epochs_num_input}_"
            f"Serv_ratio_{experiment_config.server_split_ratio}_"
            f"Num_Class_{experiment_config.num_classes}_"
            f"same_Clients_{experiment_config.identical_clients}_"
            f"with_s_net{experiment_config.with_server_net}_"
            f"s_net{experiment_config.server_net_type.name}_"
            f"c_net{experiment_config.client_net_type.name}_"

        )

def create_record_data(clients, server):
    records_dict={"clients":{},"server":{}}



    client_loss_measures = {}
    client_accuracy_test_measures = {}
    client_accuracy_pl_measures = {}
    client_accuracy_test_measures_k_half_cluster = {}
    client_accuracy_pl_measures_k_half_cluster = {}
    for client in clients:
        client_loss_measures[client.id_]=client.loss_measures
        client_accuracy_test_measures[client.id_]=client.accuracy_test_measures
        client_accuracy_pl_measures[client.id_]=client.accuracy_pl_measures
        client_accuracy_test_measures_k_half_cluster[client.id_] = client.accuracy_test_measures_k_half_cluster
        client_accuracy_pl_measures_k_half_cluster[client.id_] = client.accuracy_pl_measures_k_half_cluster
    records_dict["clients"]["loss"] = client_loss_measures
    records_dict["clients"]["accuracy_test"] = client_accuracy_test_measures
    records_dict["clients"]["accuracy_test_server_data"] = client_accuracy_pl_measures
    records_dict["clients"]["accuracy_test(k=c/2)"] = client_accuracy_test_measures_k_half_cluster
    records_dict["clients"]["accuracy_test_server_data(k=c/2)"] = client_accuracy_pl_measures_k_half_cluster

    server_loss_measures = {}
    server_accuracy_test_measures = {}
    server_accuracy_pl_measures = {}
    server_accuracy_test_measures_k_half_cluster = {}
    server_accuracy_pl_measures_k_half_cluster = {}
    for server_cluster_id,model in server.model_dict:
        server_loss_measures[server_cluster_id] = server.loss_measures
        server_accuracy_test_measures[server_cluster_id] = server.accuracy_test_measures
        server_accuracy_pl_measures[server_cluster_id] = server.accuracy_pl_measures
        server_accuracy_test_measures_k_half_cluster[server_cluster_id] = server.accuracy_test_measures_k_half_cluster
        server_accuracy_pl_measures_k_half_cluster[server_cluster_id] = server.accuracy_pl_measures_k_half_cluster



    records_dict["server"]["loss"] = server_loss_measures
    records_dict["server"]["accuracy_test"] = server_accuracy_test_measures
    records_dict["server"]["accuracy_test_server_data"] = server_accuracy_pl_measures
    records_dict["server"]["accuracy_test(k=c/2)"] = server_accuracy_test_measures_k_half_cluster
    records_dict["server"]["accuracy_test_server_data(k=c/2)"] = server_accuracy_pl_measures_k_half_cluster


    return RecordData(records_dict)






if __name__ == '__main__':
    print(device)




    exp_type = ExpType.short


    data_types =[DataType.IID,DataType.NonIID]

    nets_types_list  = [NetsType.C_alex_S_alex,NetsType.C_alex_S_vgg,NetsType.C_alex_S_None]
    cluster_architectures_list = [ClusterArchitecture.KMeans_same_output_per_cluster]
    num_cluster_list = [2]#[1,5,10]


    for data_type in data_types:
        data_to_pickle = {}
        for num_cluster in num_cluster_list:
            experiment_config.num_clusters = num_cluster
            data_to_pickle[num_cluster] = {}
            for net_type in nets_types_list:
                data_to_pickle[num_cluster][net_type.name] = {}
                for single_cluster_architecture in cluster_architectures_list:
                    data_to_pickle[num_cluster][net_type.name][single_cluster_architecture.name] = []

                    experiment_config.update_type_of_experiment(net_type,data_type,single_cluster_architecture,exp_type)
                    torch.manual_seed(experiment_config.seed_num)
                    clients_data_dict, server_data, test_set = create_data()
                    clients,clients_ids = create_clients(clients_data_dict,server_data,test_set)
                    server = Server(id_="server",global_data=server_data,test_data = test_set,clients_ids = clients_ids)

                    for t in range(experiment_config.iterations):
                        print("----------------------------iter number:"+str(t))
                        for c in clients:
                            c.iterate(t)
                            print()
                        for c in clients:
                            server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
                        server.iterate(t)
                        for c in clients:
                            c.pseudo_label_received = server.pseudo_label_to_send[c.id_]
                        rd = create_record_data(clients, server)
                        data_to_pickle[num_cluster][net_type.name][single_cluster_architecture.name] = rd
                    #create_pickle(clients,server)



                        pik_name = data_type.name
                        pickle_file_path = pik_name + ".pkl"

                        with open(pickle_file_path, "wb") as file:
                            pickle.dump(data_to_pickle, file)
