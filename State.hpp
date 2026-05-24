#pragma once
#include <vector>
#include <cmath>
#include <array>

constexpr size_t MAX_DISTANCE = 1500;
constexpr size_t MAX_VELOCITY = 8;
constexpr size_t DISTANCE_COUNT = 8;
constexpr size_t PARAM_COUNT = DISTANCE_COUNT + 13;

struct State {
	State() = default;
	virtual ~State() = default;
	virtual std::vector<double> serialise() const;		//Returns serialized State data in a form of std::vector<double>
}; 

struct SpeedwayState : State {
	std::array<double, DISTANCE_COUNT> distances;		//Set of distances from the player to the boards at different angles
	std::array<double, 2> velocity;						//Player X and Y velocities
	std::array<double, 2> absAngle;						//Sin and cos of the absolute player angle (angle relative to the screen)
	double checkpointDistance;							//Distance from the player to the checkpoint
	std::array<double, 2> checkpointAngle;				//Sin and cos of the angle at which the checkpoint-player distance is the smallest
	double playerDistance;								//Distance from the player to the nearest other player
	std::array<double, 2> playerAngle;					//Sin and cos of the angle between the player and the nearest other player
	bool isHittingBoard;								//Indicator of the player hitting the boards
	bool isHittingPlayer;								//Indicator of the player hitting other player
	bool isPassingCheckpoint;							//Indicator of the player passing a checkpoint			

	SpeedwayState(
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
	);
	std::vector<double> serialise() const override;		//Returns serialized State data in a form of std::vector<double>
};