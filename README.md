# 具身智能 / 机器人学习笔记

记录我在机器人仿真与强化学习方向的学习过程。

## 目录

| 笔记 | 说明 |
|------|------|
| [legged_gym_学习笔记.md](./legged_gym_学习笔记.md) | legged_gym + Isaac Gym 代码精读：从 RL 概念到逐个文件读源码，含跑通与排错记录 |

## 环境说明

本项目笔记基于以下环境整理：

- **仿真器**：Isaac Gym Preview 4
- **训练框架**：legged_gym（ETH RSL）
- **算法库**：rsl_rl（PPO）
- **Python**：3.8 / PyTorch 2.4.1 + CUDA 12.1

## 说明

笔记中的行号、参数值、关节限幅均来自真实代码（`legged_gym` + `a1.urdf`），不是通用模板，仅供参考。
