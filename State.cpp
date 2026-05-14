#include "State.hpp"
std::vector<double> State::vector() const {
	std::vector<double> stateVec{};
	for (size_t i{}; i < distances.size(); i++)
		stateVec.push_back(distances[i]);
	for (size_t i{}; i < absAngle.size(); i++)
		stateVec.push_back(absAngle[i]);
	stateVec.push_back(forwardVel);
	stateVec.push_back(sidewaysVel);
	stateVec.push_back(checkpointDistance);
	stateVec.push_back(static_cast<double>(isHitting));
	return stateVec;
}