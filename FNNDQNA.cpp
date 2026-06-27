#include "FNNDQNA.hpp"

FNNDQNA::FNNDQNA(
	const DQNHyperParams& dqnparams,
	std::shared_ptr<FNN> main,
	std::shared_ptr<FNN> target
): params(dqnparams), main(std::move(main)), target(std::move(target)) {}

DQNHyperParams FNNDQNA::getParams() const { return params; }

void FNNDQNA::setParams(const DQNHyperParams& params) {	this->params = params; }
void FNNDQNA::updateNNs(std::shared_ptr<FNN> main, std::shared_ptr<FNN> target) {
	this->main = main;
	this->target = target;
}

std::shared_ptr<FNN> FNNDQNA::getTargetNN() const { return target; }

size_t FNNDQNA::act(const DQNState& state, bool getConsoleOutput) {
	return act(state.serialise(), getConsoleOutput);
}

size_t FNNDQNA::act(const std::vector<double>& stateVec, bool getConsoleOutput) {
	double rand = mfuncs::getRandomDouble(0.0, 1.0);
	if (rand <= params.epsilon)
		return static_cast<size_t>(mfuncs::getRandomInteger(0, params.actionsCount - 1));
	main->process(stateVec);
	std::vector<double> qValues = main->getOutput();
	size_t action = mfuncs::maxIndex(qValues);
	if (getConsoleOutput) {
		std::cout << "Epsilon: " << params.epsilon;
		for (size_t i{}; i < params.actionsCount; i++) std::cout << " | Q_" << i << ": " << qValues[i];
		std::cout << " | ACTION: " << action << "\n";
	}
	return action;
}

void FNNDQNA::remember(
	const std::vector<double>& state, 
	size_t action, 
	double reward,
	const std::vector<double>& nextState, 
	bool finished
) {

	memory.push_back({ state, action, reward, nextState, finished });

	if (memory.size() > params.memoryCapacity) {
		memory.pop_front();
	}
}

void FNNDQNA::replayLearn() { // Replay previous experiences and learn on the mistakes
	if (memory.size() < params.batchSize) return; //if memory is smaller than the minimum amount to learn, return. 
	auto batch = getMemoryBatch(); // get random batch of previous memories

	for (const Transition& transition : batch) {
		double targetValue = getTargetValue(transition);

		auto currentQValues = getCurrentQValues(transition.originalState);
		currentQValues[transition.action] = targetValue;

		FNNBackpropA::trainFNNStep(main, transition.originalState, currentQValues, params.learningFactor, true);
	}
	if (params.epsilon > params.epsilonMin) params.epsilon *= params.epsilonDecayRate;
}

std::vector<Transition> FNNDQNA::getMemoryBatch() {
	std::vector<Transition> batch;
	auto indexVec = mfuncs::getNRandomUniqueIntegers(0, memory.size(), params.batchSize);
	for (auto index : indexVec)
		batch.push_back(memory[index]);
	return batch;
}

void FNNDQNA::updateTargetNN() {
	target->cloneWeights(main);
}

void FNNDQNA::supervisedLearning(const std::vector<DemonstrativeTransition>& demonstrations) {
	for (int epoch = 0; epoch < SUPERVISED_LEARNING_EPOCH_COUNT; epoch++) {
		for (const auto& transition : demonstrations) {

			std::vector<double> currentQ = getCurrentQValues(transition.state);

			for (size_t i{}; i < params.actionsCount; i++) {
				if (i == transition.action) currentQ[i] = 150.0;
				else currentQ[i] = -100.0;
			}

			FNNBackpropA::trainFNNStep(main, transition.state, currentQ, 0.001, true);
		}
	}

	updateTargetNN();
	std::cout << "Klonowanie zakonczone.\n";
}

double FNNDQNA::getTargetValue(const Transition& transition) {
	double targetValue = transition.reward;
	if (!transition.finished) {
		//Double DQN
		main->process(transition.nextState);
		std::vector<double> mainNextQValues = main->getOutput();
		size_t bestActionNextState = mfuncs::maxIndex(mainNextQValues);

		target->process(transition.nextState);
		std::vector<double> nextQValues = target->getOutput();

		//double maxNextQ = mfuncs::max(nextQValues);
		double maxNextQ = nextQValues[bestActionNextState];
		targetValue += params.gamma * maxNextQ;
	}
	return targetValue;
}

std::vector<double> FNNDQNA::getCurrentQValues(const std::vector<double>& inputVec) {
	main->process(inputVec);
	return main->getOutput();
}