# xArm6 基于世界模型的 CEM-MPC 规划与控制

[English](README_EN.md)

本项目基于 ROS2、PyTorch 和 xArm6 机械臂，实现了一个从数据采集、世界模型训练，到基于交叉熵方法（Cross-Entropy Method，CEM）的模型预测控制（Model Predictive Control，MPC）的完整流程。

项目当前版本主要用于验证：

> 能否通过学习到的动力学模型预测机械臂未来状态，并利用模型进行动作搜索和闭环控制。

---

## 1. 项目流程

```text
xArm6 ROS2 仿真
        ↓
自动采集机械臂轨迹
        ↓
构建状态转移数据集
        ↓
训练世界模型
        ↓
多步未来状态预测
        ↓
CEM 搜索未来动作序列
        ↓
MPC 在线重规划
        ↓
xArm6 闭环控制