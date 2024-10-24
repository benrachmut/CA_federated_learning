import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from config import *
from functions import *
from entities import *




if __name__ == '__main__':
    for client_split_ratio in client_split_ratio_list:

        print("client_split_ratio:",client_split_ratio)
        train_set, test_set = create_data()
        client_data_sets,server_data = split_clients_server_data(train_set,client_split_ratio)
        clients,clients_ids = create_clients(client_data_sets,server_data,test_set)
        server = Server(server_data,clients_ids)

        for t in range(iterations):
            print("----------------------------iter number:"+str(t))
            for c in clients: c.iterate(t,client_split_ratio)
            for c in clients: server.receive_single_pseudo_label(c.id_,c.pseudo_label_to_send)
            server.iterate(t,client_split_ratio)
            for c in clients: c.pseudo_label_received = server.pseudo_label_to_send

            file_name= get_file_name(round(1-client_split_ratio,2))+"_test_avg_loss"
            average_loss_df = create_mean_df(clients,file_name)
            plot_average_loss(average_loss_df=average_loss_df,filename = file_name)

        # Now compute the average test loss across clients



