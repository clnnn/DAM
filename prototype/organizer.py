from sklearn.model_selection import train_test_split
import pandas as pd
from itsdangerous import URLSafeSerializer
import numpy as np
import torch

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class Organizer:

    def __init__(self, account_index):
        self.account_index = account_index
        self.secret = "my-secret"

    def read_dataset(self, path):
        df = pd.read_csv(path)
        cols = ['Rainfall', 'Humidity3pm',
                'Pressure9am', 'RainToday', 'RainTomorrow']
        df = df[cols]
        return df

    def split_dataset(self, df):
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=RANDOM_SEED)
        return df_train, df_test

    def save_training_data(self, df_train, output_path):
        df_train.to_csv(output_path, index=False)

    def save_encrypted_test_data(self, df_test, output_path):
        serializer = URLSafeSerializer(self.secret)
        df_test = df_test.applymap(lambda x: serializer.dumps(x))
        df_test.to_csv(output_path, index=False)
    
    def reveal_decryption_key(self, contract_api):
        contract_api.reveal_decryption_key(self.account_index, self.secret)
    
    def trigger_evaluation(self, contract_api):
        contract_api.evaluate_models(self.account_index)
