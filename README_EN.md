# xArm6 World Model CEM-MPC

[中文](README.md)

A world-model-based planning and closed-loop control project for the xArm6 robot using ROS2, PyTorch, the Cross-Entropy Method (CEM), and Model Predictive Control (MPC).

## Overview

The current pipeline is:

```text
xArm6 ROS2 Simulation
        ↓
Trajectory Data Collection
        ↓
Learned Dynamics Model
        ↓
Multi-step Prediction
        ↓
CEM Planning
        ↓
Model Predictive Control
        ↓
Closed-loop xArm6 Control