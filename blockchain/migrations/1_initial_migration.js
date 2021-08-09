const Competition = artifacts.require('Competition');

module.exports = function (deployer, network, accounts) {
  const organizerAccount = accounts[0];
  const evaluatorAccount = accounts[3];
  const reward = 10e18;
  const trainingDataUrl = 'D://UDMLC//out//weatherAUS_train.csv';
  const testEncryptedDataUrl = 'D://UDMLC//out//weatherAUS_test.csv';

  deployer.deploy(
    Competition,
    evaluatorAccount,
    web3.utils.asciiToHex(trainingDataUrl),
    web3.utils.asciiToHex(testEncryptedDataUrl),
    {
      from: organizerAccount,
      value: reward,
    }
  );
};
