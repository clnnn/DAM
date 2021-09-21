import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from web3 import contract
from neural_network import Net, calculate_accuracy
from torch import nn, optim


FLOAT_PRECISION = 1000000
COMMIT_PERIOD_IN_SECONDS = 70


class Competitor:

    def __init__(self, account_index):
        self.account_index = account_index

    def read_training_data(self, contract_api):
        url = contract_api.read_training_data_url(self.account_index)
        df = pd.read_csv(url)
        return df

    def preprocess_data(self, df):
        cols = ['Rainfall', 'Humidity3pm',
                'Pressure9am', 'RainToday', 'RainTomorrow']
        df = df[cols]
        df['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
        df['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
        df = df.dropna(how='any')
        return df

    def train_data(self, df, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Select features & prediction output
        X = df[['Rainfall', 'Humidity3pm', 'RainToday', 'Pressure9am']]
        y = df[['RainTomorrow']]

        # Split data into train & test datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed)

        # Convert data from numpy to torch format
        X_train = torch.from_numpy(X_train.to_numpy()).float()
        y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
        X_test = torch.from_numpy(X_test.to_numpy()).float()
        y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

        # Create Neural Network
        net = Net(X_train.shape[1])  # in our case is 4

        # Prepare training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        # Select training device (CPU/GPU Cuda)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        net = net.to(device)
        criterion = criterion.to(device)

        # Start training
        for epoch in range(1000):
            y_pred = net(X_train)
            y_pred = torch.squeeze(y_pred)

            train_loss = criterion(y_pred, y_train)

            if epoch % 100 == 0:
                train_acc = calculate_accuracy(y_train, y_pred)

                y_test_pred = net(X_test)
                y_test_pred = torch.squeeze(y_test_pred)

                test_loss = criterion(y_test_pred, y_test)

                test_acc = calculate_accuracy(y_test, y_test_pred)
                print(
                    f'''epoch {epoch}
                    Train set - loss: {train_loss}, accuracy: {train_acc}
                    Test  set - loss: {test_loss}, accuracy: {test_acc}
                ''')

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        print(f'''[Account: {self.account_index}] Model's state_dict:''')
        for param_tensor in net.state_dict():
            print(param_tensor, "\t", net.state_dict()[param_tensor])
        return net

    def flatten_network(self, net):
        weights = []
        biases = []
        for param_tensor in net.state_dict():
            current_tensor = net.state_dict()[param_tensor].flatten()
            current_list = current_tensor.tolist()
            if param_tensor.endswith("weight"):
                weights.extend(current_list)
            elif param_tensor.endswith("bias"):
                biases.extend(current_list)

        # Scale weights & biases and convert them into integers (Reason: Solidity does not support float numbers)
        weights = list(map(lambda x: int(x * FLOAT_PRECISION), weights))
        biases = list(map(lambda x: int(x * FLOAT_PRECISION), biases))
        return weights, biases

    def commit_model(self, weights, contract_api):
        contract_api.commit_model(self.account_index, weights)

    def reveal_model(self, n_features, n_outputs, n_hidden, weights, biases, contract_api):
        contract_api.reveal_model(
            self.account_index, n_features, n_outputs, n_hidden, weights, biases)
    
    def claim_reward(self, contract_api):
        contract_api.claim_reward(self.account_index)
