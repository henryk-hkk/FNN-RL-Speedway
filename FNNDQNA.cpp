#include "FNNDQNA.hpp"

FNNDQNA::FNNDQNA(
	const DQNHyperParams& dqnparams,
	FNN& main,
	FNN& target
): params(dqnparams), main(main), target(target) {}

size_t FNNDQNA::act(const State& state) {
	double rand = mfuncs::getRandomDouble(0.0, 1.0);
	if (rand <= params.epsilon)
		return static_cast<size_t>(mfuncs::getRandomInteger(0, params.actionsCount - 1));
	main.process(state.serialise());
	std::vector<double> qValues = main.getOutput();
	return mfuncs::maxIndex(qValues);

}

void FNNDQNA::remember(const std::vector<double>& state, size_t action, double reward,
	const std::vector<double>& nextState, bool finished) {

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

		FNNBackpropA::trainFNNStep(main, transition.originalState, currentQValues, learningFactor);
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
	target.cloneWeights(main);
}

double FNNDQNA::getTargetValue(const Transition& transition) {
	double targetValue = transition.reward;
	if (!transition.finished) {
		target.process(transition.nextState);
		std::vector<double> nextQValues = target.getOutput();
		double maxNextQ = mfuncs::max(nextQValues);
		targetValue += params.gamma * maxNextQ;
	}
	return targetValue;
}

std::vector<double> FNNDQNA::getCurrentQValues(const std::vector<double>& inputVec) {
	main.process(inputVec);
	return main.getOutput();
}

double calculateReward(const SpeedwayState& stateBefore, size_t action, const SpeedwayState& stateAfter) {
	double reward = 0.0;

	if (stateAfter.isHittingBoard) reward -= 2.0;
	if (stateAfter.isHittingPlayer)	reward -= 4.0;

	if (stateAfter.isPassingCheckpoint) reward += 50.0;

	double distanceImprovement = stateBefore.checkpointDistance - stateAfter.checkpointDistance;
	reward += distanceImprovement * 50.0;

	reward -= 0.05; //existential penalty. time is passing, agent must be quick!!

	//if (action == 1 && stateAfter.isHittingBoard == false) reward -= 0.05;

	return reward;
}