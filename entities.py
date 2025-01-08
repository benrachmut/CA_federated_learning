import numpy as np
import torch
import torchvision
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader
from config import *
import torch.nn.functional as F

from abc import ABC, abstractmethod

# Define AlexNet for clients

def get_file_name(server_split_ratio):
    return \
        f"data_server_{round(server_split_ratio,ndigits=2):.1f},data_use_"+str(experiment_config.percent_train_data_use)+"_with_server_net_"+str(experiment_config.with_server_net)+"_epoch_num_"+str(experiment_config.epochs_num_input)


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define VGG16 for server
class VGGServer(nn.Module):
    def __init__(self, num_classes):
        super(VGGServer, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=None)  # No pre-trained weights
        self.vgg.classifier[6] = nn.Linear(4096, num_classes)  # Adjust for CIFAR-10

    def forward(self, x):
        # Resize input to match VGG's expected input size
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.vgg(x)
        return x


def get_client_model():
    if experiment_config.client_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=experiment_config.num_classes).to(device)
    if experiment_config.client_net_type == NetType.VGG:
        return VGGServer(num_classes=experiment_config.num_classes).to(device)


def get_server_model():
    if experiment_config.server_net_type == NetType.ALEXNET:
        return AlexNet(num_classes=experiment_config.num_classes).to(device)
    if experiment_config.server_net_type == NetType.VGG:
        return VGGServer(num_classes=experiment_config.num_classes).to(device)




class LearningEntity(ABC):
    def __init__(self,id_,global_data,test_data):
        self.global_data = global_data
        self.pseudo_label_received = None
        self.pseudo_label_to_send = None
        self.current_iteration = 0
        self.epoch_count = 0
        self.id_ = id_
        self.test_set= test_data
        self.model=None
        self.weights = None
        self.loss_measures = {}
        self.accuracy_test_measures = {}
        self.accuracy_pl_measures= {}


    def initialize_weights(self, layer):
        """Initialize weights for the model layers."""
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    #def set_weights(self):
        #if self.weights  is None:
        #    self.model.apply(self.initialize_weights)
        #else:
        #    self.model.apply(self.weights)

    def iterate(self,t):
        if experiment_config.with_weight_memory ==False:
            self.model.apply(self.initialize_weights)
        #self.set_weights()
        torch.manual_seed(self.num+t*17)
        torch.cuda.manual_seed(self.num+t*17)
        self.iteration_context(t)
        if isinstance(self,Client) or (isinstance(self,Server) and experiment_config.with_server_net):
            self.loss_measures[t]=self.evaluate_test_loss()
            self.accuracy_test_measures[t]=self.evaluate_accuracy(self.test_set)
            self.accuracy_pl_measures[t]=self.evaluate_accuracy(self.global_data)

    @abstractmethod
    def iteration_context(self,t):
        pass



    def evaluate(self):
        print("*** Generating Pseudo-Labels with Probabilities ***")

        # Create a DataLoader for the global data
        global_data_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False)

        self.model.eval()  # Set the model to evaluation mode

        all_probs = []  # List to store the softmax probabilities
        with torch.no_grad():  # Disable gradient computation
            for inputs, _ in global_data_loader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)  # Forward pass

                # Apply softmax to get the class probabilities
                probs = F.softmax(outputs, dim=1)  # Apply softmax along the class dimension

                all_probs.append(probs.cpu())  # Store the probabilities on CPU

        # Concatenate all probabilities into a single tensor
        all_probs = torch.cat(all_probs, dim=0)

        return all_probs

    def evaluate_accuracy(self,data_):
        """
           Evaluate the accuracy of the model on the given test dataset.

           Args:
               model (torch.nn.Module): The model to evaluate.
               test_data (torch.utils.data.Dataset): The test dataset.
               batch_size (int): The batch size for loading the data.
               device (str): The device to run the model on, e.g., 'cuda' or 'cpu'.

           Returns:
               float: The accuracy of the model on the test dataset.
           """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0  # To count the correct predictions
        total = 0  # To count the total predictions

        test_loader = DataLoader(data_, batch_size=experiment_config.batch_size, shuffle=False)

        with torch.no_grad():  # No need to track gradients during evaluation
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = self.model(inputs)

                # Get the predicted class
                _, predicted = torch.max(outputs, 1)

                # Update the total number of predictions and correct predictions
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = 100 * correct / total  # Calculate accuracy as a percentage
        print("accuracy is",str(accuracy))
        return accuracy

    def evaluate_test_loss(self):
        """Evaluate the model on the test set and return the loss."""
        self.model.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(self.test_set, batch_size=experiment_config.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()  # Define the loss function
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # Move to device

                # Get the model outputs
                outputs = self.model(inputs)
                # Calculate loss
                loss = criterion(outputs, targets)

                # Accumulate the loss
                total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
                total_samples += inputs.size(0)  # Count the number of samples
        ans = total_loss / total_samples if total_samples > 0 else float('inf')
        print(f"Iteration [{self.current_iteration}], Test Loss: {ans:.4f}")

        # Return average loss
        return ans  # Avoid division by zero

    def train(self,mean_pseudo_labels):


        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_learning_rate)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue  # Skip the rest of the loop for this batch

                # Check for NaN or Inf in pseudo targets
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss
                loss = criterion(outputs_prob, pseudo_targets)

                # Check if the loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train}], Loss: {avg_loss:.4f}")

        self.weights =self.model.state_dict()
        return avg_loss


class Client(LearningEntity):
    def __init__(self, id_, client_data, global_data,test_data,class_):
        LearningEntity.__init__(self,id_,global_data,test_data)
        self.num = (self.id_+1)*17
        self.local_data = client_data
        self.class_ = class_
        self.epoch_count = 0
        self.model = get_client_model()
        self.model.apply(self.initialize_weights)
        self.train_learning_rate = experiment_config.learning_rate_train_c
        self.weights = None
        self.global_data =global_data


    def iteration_context(self, t):
        self.current_iteration = t
        if t>0:
            train_loss = self.train(self.pseudo_label_received)
        train_loss = self.fine_tune()
        self.pseudo_label_to_send = self.evaluate()
        print()
        #test_loss = self.evaluate_test_loss()

    def __str__(self):
        return "Client " + str(self.id_)

    def fine_tune(self):
        print("*** " + self.__str__() + " fine-tune ***")

        # Load the weights into the model
        #if self.weights is  None:
        #    self.model.apply(self.initialize_weights)
        #else:
        #    self.model.load_state_dict(self.weights)

        # Create a DataLoader for the local data
        fine_tune_loader = DataLoader(self.local_data, batch_size=experiment_config.batch_size, shuffle=True)
        self.model.train()  # Set the model to training mode

        # Define loss function and optimizer

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate)

        for epoch in range(experiment_config.epochs_num_input):
            self.epoch_count += 1
            epoch_loss = 0
            for inputs, targets in fine_tune_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = criterion(outputs, targets)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            result_to_print = epoch_loss / len(fine_tune_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_input}], Loss: {result_to_print:.4f}")
        #self.weights = self.model.state_dict()self.weights = self.model.state_dict()

        return  result_to_print



class Server(LearningEntity):
    def __init__(self,id_,global_data,test_data, clients_ids):
        LearningEntity.__init__(self, id_,global_data,test_data)
        self.num = (1000)*17

        self.received_pseudo_labels = {}
        self.clients_ids = clients_ids
        self.reset_clients_received_pl()
        self.model = get_server_model()
        self.model.apply(self.initialize_weights)
        self.train_learning_rate = experiment_config.learning_rate_train_s

        self.weights = None





    def receive_single_pseudo_label(self, sender, info):
        self.received_pseudo_labels[sender] = info


    def iteration_context(self,t):
        self.current_iteration = t
        self.server_split_ratio = experiment_config.server_split_ratio
        if experiment_config.with_server_net:
            mean_pseudo_labels = self.get_mean_pseudo_labels()  # #

            self.train(mean_pseudo_labels)
            self.pseudo_label_to_send = self.evaluate()


        else:
            self.pseudo_label_to_send = self.get_mean_pseudo_labels()
        print()
        self.reset_clients_received_pl()





    def reset_clients_received_pl(self):
        for id_ in self.clients_ids:
            self.received_pseudo_labels[id_] = None


    def k_means_grouping(self, k):
        """
        Groups agents into k clusters based on the similarity of their pseudo-labels.

        Args:
            k (int): Number of clusters for k-means.

        Returns:
            dict: A dictionary where keys are cluster indices (0 to k-1) and values are lists of client IDs in that cluster.
        """
        # Use self.received_pseudo_labels as client data
        client_data = self.received_pseudo_labels

        # Extract client IDs and pseudo-labels
        client_ids = list(client_data.keys())
        pseudo_labels = [client_data[client_id].flatten().numpy() for client_id in client_ids]

        # Stack pseudo-labels into a matrix for clustering
        data_matrix = np.vstack(pseudo_labels)

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_assignments = kmeans.fit_predict(data_matrix)

        # Group client IDs by cluster
        clusters = {i: [] for i in range(k)}
        for client_id, cluster in zip(client_ids, cluster_assignments):
            clusters[cluster].append(client_id)

        return clusters
    def get_mean_pseudo_labels(self):
        # Stack the pseudo labels tensors into a single tensor
        if experiment_config.num_clusters>1:
            clusters = self.k_means_grouping(experiment_config.num_clusters)
            stop here! I have the clusters by client id
        pseudo_labels_list = list(self.received_pseudo_labels.values())

        stacked_labels = torch.stack(pseudo_labels_list)

        # Average the pseudo labels across clients
        average_pseudo_labels = torch.mean(stacked_labels, dim=0)

        return average_pseudo_labels

    def train(self,mean_pseudo_labels):


        print(f"*** {self.__str__()} train ***")
        server_loader = DataLoader(self.global_data, batch_size=experiment_config.batch_size, shuffle=False, num_workers=4,
                                   drop_last=True)

        self.model.train()
        criterion = nn.KLDivLoss(reduction='batchmean')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=experiment_config.learning_rate_train_s)

        pseudo_targets_all = mean_pseudo_labels.to(device)

        for epoch in range(experiment_config.epochs_num_train):
            self.epoch_count += 1
            epoch_loss = 0

            for batch_idx, (inputs, _) in enumerate(server_loader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = self.model(inputs)
                # Check for NaN or Inf in outputs

                # Convert model outputs to log probabilities
                outputs_prob = F.log_softmax(outputs, dim=1)
                # Slice pseudo_targets to match the input batch size
                start_idx = batch_idx * experiment_config.batch_size
                end_idx = start_idx + inputs.size(0)
                pseudo_targets = pseudo_targets_all[start_idx:end_idx].to(device)

                # Check if pseudo_targets size matches the input batch size
                if pseudo_targets.size(0) != inputs.size(0):
                    print(
                        f"Skipping batch {batch_idx}: Expected pseudo target size {inputs.size(0)}, got {pseudo_targets.size(0)}")
                    continue  # Skip the rest of the loop for this batch

                # Check for NaN or Inf in pseudo targets
                if torch.isnan(pseudo_targets).any() or torch.isinf(pseudo_targets).any():
                    print(f"NaN or Inf found in pseudo targets at batch {batch_idx}: {pseudo_targets}")
                    continue

                # Normalize pseudo targets to sum to 1
                pseudo_targets = F.softmax(pseudo_targets, dim=1)

                # Calculate the loss
                loss = criterion(outputs_prob, pseudo_targets)

                # Check if the loss is NaN or Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf loss encountered at batch {batch_idx}: {loss}")
                    continue

                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(server_loader)
            print(f"Epoch [{epoch + 1}/{experiment_config.epochs_num_train}], Loss: {avg_loss:.4f}")

        self.weights =self.model.state_dict()
        return avg_loss


    def __str__(self):
        return "server"





