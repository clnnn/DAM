from web3 import Web3
import json

HTTP_PROVIDER_URL = "http://127.0.0.1:7545"
COMPILED_CONTRACT_PATH = "../blockchain/build/contracts/Competition.json"


class ContractAPI:
    def __init__(self, contract_address):
        self.w3 = Web3(Web3.HTTPProvider(HTTP_PROVIDER_URL))
        with open(COMPILED_CONTRACT_PATH) as file:
            contract_json = json.load(file)
            contract_abi = contract_json['abi']
        self.contract = self.w3.eth.contract(
            address=contract_address, abi=contract_abi)

    def read_training_data_url(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        training_data_url = Web3.toText(
            self.contract.functions.trainingDataUrl().call())
        training_data_url = training_data_url.replace("//", "/")
        return training_data_url

    def commit_model(self, account_index, weights):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        hashed_weights = Web3.solidityKeccak(["int256[]"], [weights])
        self.contract.functions.commitModel(
            Web3.toBytes(hashed_weights)).transact()

    def reveal_model(self, account_index, n_features, n_outputs, n_hidden, weights, biases):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        self.contract.functions.revealModel(
            n_features, n_outputs, n_hidden, weights, biases).transact()

    def reveal_decryption_key(self, account_index, secret):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        self.contract.functions.revealDecryptionKey(secret).transact()

    def get_evaluate_models_event_filter(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        return self.contract.events.EvaluateModels.createFilter(fromBlock="latest")

    def evaluate_models(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        self.contract.functions.evaluateModels().transact()

    def get_submissions_count(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        return self.contract.functions.getSubmissionsCount().call()

    def get_submitted_model_by_index(self, account_index, model_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        return self.contract.functions.getSubmittedModelByIndex(model_index).call()

    def get_test_data_url(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        test_data_url = Web3.toText(
            self.contract.functions.testEncryptedDataUrl().call())
        test_data_url = test_data_url.replace("//", "/")
        return test_data_url

    def get_revealed_secret(self, account_index):
        self.w3.eth.default_account = self.w3.eth.accounts[account_index]
        return self.contract.functions.decryptionKey().call()

    def execute_evaluation_callback(self, account_index, accuracy_per_index):
        self.contract.functions.evaluationCallback(accuracy_per_index).transact({
            "from": self.w3.eth.accounts[account_index]
        })

    def claim_reward(self, account_index):
        self.contract.functions.claimReward().transact({
            "from": self.w3.eth.accounts[account_index]
        })
