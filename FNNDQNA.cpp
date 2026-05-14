#include "FNNDQNA.hpp"

FNNDQNA::FNNDQNA(const DQNHyperParams& dqnparams, FNN& main, FNN& target): params(dqnparams), main(main), target(target) {
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

void FNNDQNA::saveTransition() {
	/* get current state
	* get ai action
	* get reward
	* get next state
	* get if finished
	* Transition transition{currentState, action, reward, nextState};
	memory.push_back(transition);*/
}

void FNNDQNA::replayLearn() {
	if (memory.size() < params.batchSize) return;

	auto batch = getMemoryBatch();

	for (const Transition& transition : batch) {
		double targetValue = getTargetValue(transition);


		auto currentQValues = getCurrentQValues(transition.originalState.vector());
		currentQValues[transition.action] = targetValue;

		FNNBackpropA::trainFNNStep(main, transition.originalState.vector(), currentQValues, learningFactor);
	}
	if (params.epsilon > params.epsilonMin) params.epsilon *= params.epsilonDecayRate;
}

std::vector<Transition> FNNDQNA::getMemoryBatch() {
	std::vector<Transition> batch;
	auto indexVec = mfuncs::getNRandomUniqueIntegers(0, memory.size(), params.batchSize);
	for (size_t i{}; i < indexVec.size(); i++)
		batch.push_back(memory[indexVec[i]]);
	return batch;
}

void FNNDQNA::updateTargetNN() {
	target = FNN(main);
}

double FNNDQNA::getTargetValue(const Transition& transition) {
	double targetValue = transition.reward;
	if (!transition.finished) {
		target.process(transition.nextState.vector());
		std::vector<double> nextQValues = target.getOutput();
		double maxNextQ = mfuncs::max(nextQValues);
		targetValue += params.gamma * maxNextQ;
	}
}

std::vector<double> FNNDQNA::getCurrentQValues(const std::vector<double>& inputVec) {
	main.process(inputVec);
	return main.getOutput();
}
