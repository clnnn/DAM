// SPDX-License-Identifier: MIT
pragma solidity >=0.5.0 <0.9.0;

contract Competition {
    enum Stage {
        AcceptingCommits,
        AcceptingReveals,
        RevealingDecryptionKey,
        Evaluating,
        DeterminingWinner,
        Finished
    }

    Stage public currentStage = Stage.AcceptingCommits;
    uint256 public creationTime = block.timestamp;

    modifier atStage(Stage _stage) {
        require(currentStage == _stage);
        _;
    }

    modifier transitionAfter() {
        _;
        nextStage();
    }

    modifier timedTransitions() {
        if (
            currentStage == Stage.AcceptingCommits &&
            block.timestamp >= creationTime + 1 minutes
        ) {
            nextStage();
        }

        if (
            currentStage == Stage.AcceptingReveals &&
            block.timestamp >= creationTime + 2 minutes
        ) {
            nextStage();
        }

        if (
            currentStage == Stage.RevealingDecryptionKey &&
            block.timestamp >= creationTime + 3 minutes
        ) {
            nextStage();
        }
        _;
    }

    function nextStage() internal {
        currentStage = Stage(uint256(currentStage) + 1);
    }

    event EvaluateModels();

    struct Model {
        address participant;
        uint256 numberOfInputLayerNeurons; // e.g. 5
        uint256 numberOfOutputLayerNeurons; // e.g. 1
        uint256[] numberOfHiddenLayerNeurons; // e.g. [5, 3] - 5 for the first hidden layer and 3 for the second
        int256[] weights;
        int256[] biases;
    }

    address public organizer;
    string public trainingDataUrl;
    string public testEncryptedDataUrl;

    mapping(address => bool) public participation;
    mapping(address => bytes32) public commitedWeights;
    Model[] public submittedModels;

    address public evaluator;
    string public decryptionKey;
    int256[] public modelAccuracyPerIndex;
    uint256 public bestSubmittedModelIndex;

    constructor(
        address _evaluator,
        string memory _trainingDataUrl,
        string memory _testEncryptedDataUrl
    ) public payable {
        organizer = msg.sender;
        evaluator = _evaluator;
        trainingDataUrl = _trainingDataUrl;
        testEncryptedDataUrl = _testEncryptedDataUrl;
    }

    function commitModel(bytes32 commit)
        public
        timedTransitions
        atStage(Stage.AcceptingCommits)
    {
        require(!participation[msg.sender], "Sender already commited.");

        participation[msg.sender] = true;
        commitedWeights[msg.sender] = commit;
    }

    function revealModel(
        uint256 numberOfInputLayerNeurons,
        uint256 numberOfOutputLayerNeurons,
        uint256[] memory numberOfHiddenLayerNeurons,
        int256[] memory weights,
        int256[] memory biases
    ) public timedTransitions atStage(Stage.AcceptingReveals) {
        require(participation[msg.sender], "Sender didn't commit before.");
        require(
            keccak256(abi.encodePacked(weights)) == commitedWeights[msg.sender],
            "The commited weights are different from the revealed ones"
        );

        submittedModels.push(
            Model(
                msg.sender,
                numberOfInputLayerNeurons,
                numberOfOutputLayerNeurons,
                numberOfHiddenLayerNeurons,
                weights,
                biases
            )
        );
    }

    function revealDecryptionKey(string memory _decryptionKey)
        public
        timedTransitions
        atStage(Stage.RevealingDecryptionKey)
        transitionAfter
    {
        require(
            msg.sender == organizer,
            "Only the organizer can reveal the key."
        );
        decryptionKey = _decryptionKey;
    }

    function evaluateModels() public atStage(Stage.Evaluating) transitionAfter {
        emit EvaluateModels(); 
    }

    function evaluationCallback(int256[] memory _modelAccuracyPerIndex)
        public
        payable
        atStage(Stage.DeterminingWinner)
        transitionAfter
    {
        require(
            msg.sender == evaluator,
            "This callback is executed only by the evaluator."
        );

        modelAccuracyPerIndex = _modelAccuracyPerIndex;

        uint256 winner_index = 0;
        for (
            uint256 index = 0;
            index < modelAccuracyPerIndex.length;
            index++
        ) {
            if (
                modelAccuracyPerIndex[index] >
                modelAccuracyPerIndex[winner_index]
            ) {
                winner_index = index;
            }
        }

        bestSubmittedModelIndex = winner_index;
    }

    function claimReward() public payable atStage(Stage.Finished) {
        Model memory bestModel = submittedModels[bestSubmittedModelIndex];
        require(
            msg.sender == bestModel.participant,
            "Only the winner can claim this reward."
        );

        address payable winner = address(uint160(bestModel.participant)); // or payable(..) for >= 0.6.0
        winner.transfer(address(this).balance);
    }

    function getSubmissionsCount() public view returns (uint256) {
        return submittedModels.length;
    }

    function getSubmittedModelByIndex(uint256 index)
        public
        view
        returns (
            address,
            uint256,
            uint256,
            uint256[] memory,
            int256[] memory,
            int256[] memory
        )
    {
        return (
            submittedModels[index].participant,
            submittedModels[index].numberOfInputLayerNeurons,
            submittedModels[index].numberOfOutputLayerNeurons,
            submittedModels[index].numberOfHiddenLayerNeurons,
            submittedModels[index].weights,
            submittedModels[index].biases
        );
    }
}
