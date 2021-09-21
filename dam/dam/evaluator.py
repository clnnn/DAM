import time
import torch
from neural_network import Net, calculate_accuracy
import pandas as pd
from itsdangerous import URLSafeSerializer

FLOAT_PRECISION = 1000000


class Evaluator:
    def __init__(self, account_index):
        self.account_index = account_index
        self.evaluated = False

    def listen_for_evaluation(self, poll_interval, contract_api):
        event_filter = contract_api.get_evaluate_models_event_filter(
            self.account_index)
        while not self.evaluated:
            for event in event_filter.get_new_entries():
                self.handle_evaluation_event(contract_api)
            time.sleep(poll_interval)
    
    def compute_accuracy(self, result):
        return int((result["Correct"].sum() / len(result)) * FLOAT_PRECISION)

    def handle_evaluation_event(self, contract_api):
        secret = contract_api.get_revealed_secret(self.account_index)
        test_data_url = contract_api.get_test_data_url(self.account_index)
        df_test = pd.read_csv(test_data_url)

        serializer = URLSafeSerializer(secret)
        df_test = df_test.applymap(lambda x: serializer.loads(x))

        accuracy_per_index = []
        for model_index in range(contract_api.get_submissions_count(self.account_index)):
            model = contract_api.get_submitted_model_by_index(
                self.account_index, model_index)
            result = self.evaluate(model, df_test)
            accuracy_per_index.append(self.compute_accuracy(result))
        contract_api.execute_evaluation_callback(self.account_index, accuracy_per_index)
        self.evaluated = True

    def evaluate(self, model, df_test):
        n_inputs = model[1]
        n_outputs = model[2]
        n_hidden = model[3]
        weights = list(map(lambda x: x / FLOAT_PRECISION, model[4]))
        biases = list(map(lambda x: x / FLOAT_PRECISION, model[5]))

        # Extract tensor sizes for weights
        number_of_neurons = [n_inputs] + n_hidden + [n_outputs]  # [4, 5, 3, 1]
        tensor_sizes = []
        for first, second in zip(number_of_neurons, number_of_neurons[1:]):
            tensor_sizes.append((first, second))  # [(4, 5), (5, 3), (3, 1)]

        # Create weights tensors
        weight_tensors = []
        for tensor_size in tensor_sizes:
            size = tensor_size[0] * tensor_size[1]
            tensor_elements = weights[:size]
            weights = weights[size:]
            weight_tensor = torch.tensor(tensor_elements, dtype=torch.float32).reshape(
                (tensor_size[1], tensor_size[0]))
            weight_tensors.append(weight_tensor)

        # Create biases tensors
        biases_tensors = []
        for neurons in number_of_neurons[1:]:
            tensor_elements = biases[:neurons]
            biases = biases[neurons:]
            bias_tensor = torch.tensor(
                tensor_elements, dtype=torch.float32).reshape(neurons)
            biases_tensors.append(bias_tensor)

        # Create State dict
        state_dict = {
            'fc1.weight': weight_tensors[0],
            'fc2.weight': weight_tensors[1],
            'fc3.weight': weight_tensors[2],
            'fc1.bias': biases_tensors[0],
            'fc2.bias': biases_tensors[1],
            'fc3.bias': biases_tensors[2]
        }

        # Initilize empty Neural Network & load state dict
        net = Net(n_inputs)
        net.load_state_dict(state_dict)

        # Preprocess - (similar as competitor)
        cols = ['Rainfall', 'Humidity3pm',
                'Pressure9am', 'RainToday', 'RainTomorrow']
        df_test = df_test[cols]
        df_test['RainToday'].replace({'No': 0, 'Yes': 1}, inplace=True)
        df_test['RainTomorrow'].replace({'No': 0, 'Yes': 1}, inplace=True)
        df_test = df_test.dropna(how='any')

        # Select features & prediction output
        X_test = df_test[['Rainfall', 'Humidity3pm',
                          'RainToday', 'Pressure9am']]
        y_test = df_test[['RainTomorrow']]

        # Convert data from numpy to torch format
        X_test = torch.from_numpy(X_test.to_numpy()).float()
        y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

        # Select training device (CPU/GPU Cuda)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        net = net.to(device)

        # Evaluate model
        preds = []
        with torch.no_grad():
            for input in X_test:
                y_hat = net.forward(input)
                y_hat = torch.squeeze(y_hat)
                preds.append(y_hat.ge(.5).float().item())
        result = pd.DataFrame({"Y": y_test, "YHat": preds})
        result["Correct"] = [1 if corr == pred else 0 for corr,
                                    pred in zip(result["Y"], result["YHat"])]
        return result
