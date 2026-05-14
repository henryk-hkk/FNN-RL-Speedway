#include "FNNDQNA.hpp"

FNNDQNA::FNNDQNA(const DQNParams& dqnparams, const FNN& main, const FNN& target): params(dqnparams), main(main), target(target) {
	params = dqnparams;
}
size_t FNNDQNA::act(const State& state) {
	double rand = mfuncs::getRandomDouble(0.0, 1.0);
	if (rand <= params.epsilon)
		return static_cast<size_t>(mfuncs::getRandomInteger(0, params.actionsCount - 1));
	main.process(state.vector());
	std::vector<double> qValues = main.getOutput();
	return mfuncs::maxIndex(qValues);

}

void FNNDQNA::replayLearn() {
	if (memory.size() < params.batchSize) return;

	std::vector<Transition> batch;
	Dataset dataset;
	//generate random batch
	for (const Transition& transition : batch) {
		double targetValue = transition.reward;
		if (!transition.finished) {
			target.process(transition.nextState.vector());
			std::vector<double> nextQValues = target.getOutput();
			double maxNextQ = mfuncs::max(nextQValues);
			targetValue += params.gamma * maxNextQ;
		}
		main.process(transition.originalState.vector());
		std::vector<double> currentQValues = main.getOutput();
		currentQValues[transition.action] = targetValue;
		dataset.inputVector.push_back(transition.originalState.vector());
		dataset.desiredOutputVector.push_back(currentQValues);
	}
	FNNBackpropA::trainFNN(main, dataset, 1, 1);
	if (params.epsilon > params.epsilonMin) params.epsilon *= params.epsilonDecayRate;
}