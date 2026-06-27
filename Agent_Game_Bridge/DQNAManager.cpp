#include "DQNAManager.hpp"
#include <memory>



DQNAssets DQNAManager::initAgent() {
	std::shared_ptr<FNN> mainFNN = std::make_shared<FNN>(), targetFNN = std::make_shared<FNN>();
	mainFNN->addInputLayer(PARAM_COUNT * STATES_TO_PROCESS);
	mainFNN->addLayer(HIDDEN_LAYERS_SIZE);
	mainFNN->addLayer(HIDDEN_LAYERS_SIZE);
	mainFNN->addOutputLayer(ACTION_COUNT);
	mainFNN->setActivationFunction(AFunc::L_RELU);
	mainFNN->setOutputLayerActivationFunction(AFunc::LINEAR);

	targetFNN->addInputLayer(PARAM_COUNT * STATES_TO_PROCESS);
	targetFNN->addLayer(HIDDEN_LAYERS_SIZE);
	targetFNN->addLayer(HIDDEN_LAYERS_SIZE);
	targetFNN->addOutputLayer(ACTION_COUNT);
	targetFNN->setActivationFunction(AFunc::L_RELU);
	targetFNN->setOutputLayerActivationFunction(AFunc::LINEAR);

	targetFNN->cloneWeights(mainFNN);

	// gamma | epsilon | epsilonMin | epsilonDecayRate | learningFactor | batchSize | actionsCount | memory
	DQNHyperParams params{ 0.995, 1.0, 0.1, 0.999995, 0.001, 256, ACTION_COUNT, 60000 };

	std::shared_ptr<FNNDQNA> dqnAgent = std::make_shared<FNNDQNA>(params, mainFNN, targetFNN);

	return DQNAssets{ mainFNN, targetFNN, dqnAgent };
}

size_t DQNAssets::getAction(const std::shared_ptr<DQNState> state) {
	return dqnAgent->act(*state);
}

void DQNAssets::update(const FNN& fnn, bool clearEpsilon) {
	main->cloneWeights(fnn);
	target->cloneWeights(fnn);
	if (clearEpsilon) this->clearEpsilon();
	dqnAgent->updateNNs(main, target);
}

void DQNAssets::clearEpsilon() {
	auto params = dqnAgent->getParams();
	params.epsilon = params.epsilonMin;
	dqnAgent->setParams(params);
}