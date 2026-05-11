 # FNN-RL-Speedway [WIP]


> **Current Status:** Core Feedforward Neural Network (FNN) architecture implemented and validated against the MNIST dataset. Reinforcement Learning (RL) components and 2D physics integration are currently in active development.


## Overview

This project is a custom-built, low-level implementation of a Feedforward Neural Network written entirely in Modern C++ from scratch. The ultimate goal is to utilize this architecture as the core decision-making agent within a custom 2D Speedway simulation using Reinforcement Learning.


## Core Engineering Focus

Instead of relying on high-level ML libraries or vectorized matrix operations, this project is built around a granular, node-based Object-Oriented architecture. This approach demonstrates a deep understanding of memory management, data structures, and the raw mathematical mechanics of backpropagation at the individual neuron level:

* **Node-Based OOP Architecture:** Granular design utilizing interconnected structures (e.g., `Layer`, `Neuron`), processing forward passes and error gradients iteratively rather than through matrix abstraction.

* **Modern C++ & Memory Management:** Strict utilization of the Standard Template Library (STL) and smart pointers to manage network topology, ensuring robust lifecycle management and preventing memory leaks during complex weight updates.

* **Numerical Stability & Algorithm Design:** Custom implementation of backpropagation logic and activation functions (including numerically stable Softmax), handling mathematical edge cases and precision natively in C++.


## Validation & Results

To verify the mathematical correctness of the core architecture and the backpropagation algorithm, the model was initially trained and tested on the standard **MNIST dataset**.

* **Network Architecture:** 784 -> 16 -> 16 -> 10

* **Activation Functions:** Sigmoid (Hidden), Softmax (Output)

* **Loss Function:** Mean Squared Error

* **Accuracy Achieved:** 90% (on average)


## Future Roadmap (Reinforcement Learning)

With the core FNN validated, the next phase involves integrating the network with an RL environment:

* Prototyping Q-learning / Deep Q-Network (DQN) algorithms.

* Integrating state and action spaces for a 2D vehicle physics engine.

* Optimizing reward functions for track navigation and agent behavior. 
