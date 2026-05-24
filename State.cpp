#include "State.hpp"

std::vector<double> State::serialise() const {
	return std::vector<double>();
}

static inline double normaliseDist(double distance) {
	return (distance > MAX_DISTANCE ? 1.0 : distance / MAX_DISTANCE);
}

static inline std::array<double, DISTANCE_COUNT> normaliseDist(const std::vector<double>& distancesVec) {
	std::array<double, DISTANCE_COUNT> distancesArr{};
	for (size_t i{}; i < DISTANCE_COUNT; i++) distancesArr[i] = normaliseDist(distancesVec[i]);
	return distancesArr;
}

static inline std::array<double, 2> normaliseVel(const std::vector<double>& velVec) {
	return std::array<double,2>{ (velVec[0] / MAX_VELOCITY),(velVec[1] / MAX_VELOCITY) };
}

SpeedwayState::SpeedwayState(
	const std::vector<double>& distances,
	const std::vector<double>& velocity,
	double absAngle,
	double checkpointDistance,
	double checkpointAngle,
	double playerDistance,
	double playerAngle,
	bool isHittingBoard,
	bool isHittingPlayer,
	bool isPassingCheckpoint
) :
	distances(normaliseDist(distances)),
	velocity(normaliseVel(velocity)),
	absAngle({ std::sin(absAngle), std::cos(absAngle) }),
	checkpointDistance(normaliseDist(checkpointDistance)),
	checkpointAngle({ std::sin(checkpointAngle),std::cos(checkpointAngle) }),
	playerDistance(normaliseDist(playerDistance)),
	playerAngle({ std::sin(playerAngle), std::cos(playerAngle) }),
	isHittingBoard(isHittingBoard),
	isHittingPlayer(isHittingPlayer),
	isPassingCheckpoint(isPassingCheckpoint)
{}

std::vector<double> SpeedwayState::serialise() const {
	std::vector<double> stateVec{};
	stateVec.reserve(PARAM_COUNT);

	for (double d : distances) stateVec.push_back(d);
	for (double v : velocity) stateVec.push_back(v);
	for (double a : absAngle) stateVec.push_back(a);

	stateVec.push_back(checkpointDistance);
	for (double ca : checkpointAngle) stateVec.push_back(ca);

	stateVec.push_back(playerDistance);
	for (double pa : playerAngle) stateVec.push_back(pa);

	stateVec.push_back(static_cast<double>(isHittingBoard));
	stateVec.push_back(static_cast<double>(isHittingPlayer));
	stateVec.push_back(static_cast<double>(isPassingCheckpoint));
	return stateVec;
}