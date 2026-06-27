#pragma once
#include "../DQNState.hpp"
#include <cmath>
#include <array>

constexpr size_t MAX_DISTANCE = 300;
constexpr size_t MAX_VELOCITY = 8;
constexpr size_t DISTANCE_COUNT = 8;
constexpr size_t PARAM_COUNT = DISTANCE_COUNT + 12;
constexpr size_t ACTION_COUNT = 2;

struct SpeedwayState : DQNState {
	std::array<double, DISTANCE_COUNT> distances;		//Set of distances from the player to the boards at different angles
	std::array<double, 2> velocity;						//Player X and Y velocities
	double checkpointDistance;							//Distance from the player to the checkpoint
	std::array<double, 2> relativeCheckpointAngle;		//Sin and cos of the angle between the player and the checkpoint's center point
	double playerDistance;								//Distance from the player to the nearest other player
	std::array<double, 2> playerAngle;					//Sin and cos of the angle between the player and the nearest other player
	std::array<double, 2> slipAngle;					//Relative velocity angle		
	bool isHittingBoard;								//Indicator of the player hitting the boards
	bool isHittingPlayer;								//Indicator of the player hitting other player
	bool isPassingCheckpoint;							//Indicator of the player passing a checkpoint			

	SpeedwayState(
		const std::array<double, DISTANCE_COUNT>& distances,
		const std::array<double, 2>& velocity,
		double absAngle,
		double checkpointDistance,
		double checkpointAngle,
		double playerDistance,
		double playerAngle,
		bool isHittingBoard,
		bool isHittingPlayer,
		bool isPassingCheckpoint
	);
	std::vector<double> serialise() const override;		//Returns serialized State data in a form of std::vector<double>
};