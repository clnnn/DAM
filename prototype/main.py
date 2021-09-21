from threading import Thread

from web3 import contract
from competitor import Competitor
from organizer import Organizer
from contract_api import ContractAPI
from evaluator import Evaluator
import time

CSV_DATASET_PATH = r"../dataset/weatherAUS.csv"
CSV_TRAIN_DATASET_PATH = r"./out/weatherAUS_train.csv"
CSV_TEST_DATASET_PATH = r"./out/weatherAUS_test.csv"

organizer = Organizer(account_index=0)
df = organizer.read_dataset(CSV_DATASET_PATH)
df_train, df_test = organizer.split_dataset(df)
organizer.save_training_data(df_train, CSV_TRAIN_DATASET_PATH)
organizer.save_encrypted_test_data(df_test, CSV_TEST_DATASET_PATH)

deployed_contract_address = input(
    "Deploy smart contract then enter contract address: ")
contract_api = ContractAPI(deployed_contract_address)

competitor1 = Competitor(1)
df1 = competitor1.read_training_data(contract_api)
df1 = competitor1.preprocess_data(df1)
net1 = competitor1.train_data(df1, seed=42)
weights1, biases1 = competitor1.flatten_network(net1)
competitor1.commit_model(weights1, contract_api) 

competitor2 = Competitor(2)
df2 = competitor2.read_training_data(contract_api)
df2 = competitor2.preprocess_data(df2)
net2 = competitor2.train_data(df2, seed=32)
weights2, biases2 = competitor2.flatten_network(net2)
competitor2.commit_model(weights2, contract_api)  # ...others

# Wait until you can reveal the model after the commit period expires
time.sleep(65)

competitor1.reveal_model(4, 1, [5, 3], weights1,
                        biases1, contract_api)  # ...others
competitor2.reveal_model(4, 1, [5, 3], weights2,
                        biases2, contract_api)  # ...others
time.sleep(61)
organizer.reveal_decryption_key(contract_api)

evaluator = Evaluator(3)
evaluator_worker = Thread(target=evaluator.listen_for_evaluation, args=(1, contract_api), daemon=True)
evaluator_worker.start()

organizer.trigger_evaluation(contract_api)
evaluator_worker.join()

# Try to claim reward
competitor1.claim_reward(contract_api)
competitor2.claim_reward(contract_api)

