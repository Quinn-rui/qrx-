# legged_gym + Isaac Gym 代码精读笔记

> 面向对象：有自动化 / 控制背景、但没有系统学过强化学习、想看懂"四足机器人到底是怎么学会走路的"的本科生。
> 笔记写法：先建立认知地图 → 补齐 RL 概念 → 顺着训练流程走一遍 → 逐个文件精读 → 落地跑通与排错 → 迁移到新机器人 → 面试准备。
>
> 本仓库为 ETH RSL 的 **legged_gym**（论文 *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*, arXiv:2109.11978）。
> 笔记中的所有行号、参数值、关节限幅均来自本目录下的真实代码与 `resources/robots/a1/urdf/a1.urdf`，不是通用模板。

---

## 目录

- [0. 怎么用这份笔记（阅读路线）](#0-怎么用这份笔记阅读路线)
- [1. 认知地图：这个仓库到底是什么](#1-认知地图这个仓库到底是什么)
- [2. 读代码前必须懂的 12 个概念](#2-读代码前必须懂的-12-个概念)
- [3. 文件职责速查表](#3-文件职责速查表)
- [4. 一次训练的完整时间线](#4-一次训练的完整时间线)
- [5. 核心文件精读](#5-核心文件精读)
  - [5.1 legged_robot.py（最重要的文件）](#51-legged_robotpy最重要的文件)
  - [5.2 legged_robot_config.py 参数总表](#52-legged_robot_configpy参数总表)
  - [5.3 terrain.py 地形生成](#53-terrainpy地形生成)
  - [5.4 task_registry.py 与 helpers.py](#54-task_registrypy-与-helperspy)
  - [5.5 rsl_rl：PPO 的实现](#55-rsl_rlppo-的实现)
- [6. 训练全链路时序图](#6-训练全链路时序图)
- [7. 从零跑通 + 踩坑排错](#7-从零跑通--踩坑排错)
- [8. 迁移到自己的机器人（G1 / 天工 3.0 / Go2）](#8-迁移到自己的机器人g1--天工-30--go2)
- [9. 实习面试准备](#9-实习面试准备)
- [10. 四周学习路线](#10-四周学习路线)
- [11. 附录](#11-附录)

---

## 0. 怎么用这份笔记（阅读路线）

不要从第 1 页读到最后一页。建议**三遍读法**：

| 遍数 | 目标 | 读哪些 | 时间 |
|---|---|---|---|
| **第 1 遍：建立骨架** | 能画出"数据从哪来、到哪去" | 第 1、2、4、6 章 | 2~3 小时 |
| **第 2 遍：抠代码** | 能指出"哪一行实现了什么" | 第 3、5 章，边读边打开源文件对照 | 2~3 天 |
| **第 3 遍：动手** | 能改参数、能迁移、能调错 | 第 7、8 章 + 每次改一个参数观察结果 | 1~2 周 |

**最重要的一条建议**：不要试图读懂每一行。这个仓库 2600 行 Python，真正决定机器人行为的只有大约 200 行：

- `_compute_torques`（1 行公式，决定力矩怎么算）
- `compute_observations`（1 个 `torch.cat`，决定机器人"看到"什么）
- `_prepare_reward_function` + 十几个 `_reward_*`（决定机器人"想"做什么）
- `check_termination`（决定什么时候"摔倒重置"）
- `_parse_cfg`（决定时间尺度）
- `terrain.py` 的 `curiculum` + `make_terrain`（决定地形怎么由易到难）
- `rsl_rl/algorithms/ppo.py` 的 `update`（决定策略怎么更新）

先把这 200 行吃透，剩下的都是工程脚手架。

---

## 1. 认知地图：这个仓库到底是什么

### 1.1 三个层次

```
┌────────────────────────────────────────────────────────────────┐
│  第三层：rsl_rl/           ← "大脑"（算法层）                    │
│  PPO 算法、Actor-Critic 网络、经验回放缓冲区                     │
│  只管：obs 进来 → action 出去；reward 回来 → 更新网络权重         │
│  完全不知道什么是"四足机器人"                                     │
├────────────────────────────────────────────────────────────────┤
│  第二层：legged_gym/       ← "世界的规则"（任务层）               │
│  定义观测空间、动作空间、奖励函数、终止条件、地形、域随机化         │
│  只管：把物理仿真包装成标准 RL 接口 (obs, rew, done)              │
├────────────────────────────────────────────────────────────────┤
│  第一层：isaacgym/         ← "物理世界"（仿真层，NVIDIA 官方库）   │
│  GPU 上并行跑几千个刚体仿真、解析 URDF、接触力计算                │
│  只管：给我力矩 → 算出下一步的位置/速度/接触力                    │
└────────────────────────────────────────────────────────────────┘
```

**为什么会解耦成这样？** 因为 RL 算法研究者不该关心"机器人的脚该怎么建模"，而机器人研究者不该关心"PPO 的裁剪系数怎么调"。两层之间靠一个极窄的接口通信——就是 [rsl_rl/rsl_rl/env/vec_env.py](rsl_rl/rsl_rl/env/vec_env.py) 那个 30 行的抽象类 `VecEnv`，只有 4 个方法：`step`、`reset`、`get_observations`、`get_privileged_observations`。

如果你把这个接口实现一遍，你的任何仿真环境（MuJoCo、PyBullet、甚至真机）都能直接套用 rsl_rl 的 PPO。**这是面试里非常值得讲的一个点。**

### 1.2 关键设计思想：两种"继承"

这个仓库的全套魔法建立在两个继承机制上，理解它们等于理解了一半的代码：

**（1）配置类的继承链** —— 决定"改哪个文件能改哪台机器人"

```
BaseConfig                          # 自动把嵌套 class 实例化（base_config.py）
  └── LeggedRobotCfg                # 通用四足配置（legged_robot_config.py）
        └── A1RoughCfg              # 只覆盖 A1 特有的东西（a1_config.py）
        └── AnymalCRoughCfg         # ANYmal-C 特有（+ 执行器网络）
              └── AnymalCFlatCfg    # 再覆盖成平地版本
```

所以 [a1_config.py](legged_gym/envs/a1/a1_config.py) 只有 50 行有效代码，却能定义一整台机器人的任务——**其余 90% 都继承自 `LeggedRobotCfg`**。你要改 A1 的行为，优先改 `a1_config.py`，永远不要去动 `legged_robot_config.py`（那会污染所有机器人，README 里也专门警告了这一点）。

**（2）奖励函数的"字符串反射"** —— 决定"加一个奖励有多简单"

这是本仓库最优雅也最容易让人迷惑的设计：

```python
# legged_robot.py:544  _prepare_reward_function()
for name, scale in self.reward_scales.items():
    if name == "termination": continue
    self.reward_names.append(name)
    name = '_reward_' + name            # ★ 拼字符串
    self.reward_functions.append(getattr(self, name))   # ★ 反射取方法
```

意思是：**只要你在配置里写了一个非零的奖励名 `foo`，代码就会自动去找名叫 `_reward_foo` 的方法并调用它**。于是"加一个奖励"变成了两步：

1. 在 `cfg.rewards.scales` 里写 `my_reward = 0.5`
2. 在环境类里写 `def _reward_my_reward(self): return ...`

没有注册、没有列表、没有 if-else。代价是**名字打错不会有任何报错**——见第 5.1.7 节的 `feet_stumble` 陷阱。

### 1.3 这个仓库的"卖点"在哪（对比其他方案）

| 方案 | 并行环境数 | 训练 A1 走路耗时 | 特点 |
|---|---|---|---|
| 单环境 MuJoCo + PPO | 1~几十 | 几天 | 慢，但轻量、开源友好 |
| **legged_gym + Isaac Gym** | **几千~上万** | **几十分钟~几小时** | 靠 GPU 并行把采样瓶颈打掉 |
| Isaac Lab（官方后继者） | 几千+ | 类似 | 功能更强，但 API 更重，教程少 |
| Orbit / skrl / legged_gym forks | — | — | 生态衍生版 |

这份工作真正解决的问题不是"发明了新算法"，而是**把采样速度提升了 2~3 个数量级**：PPO 是 on-policy 算法，样本用完就扔，所以采样速度直接等于训练速度上限。当你能同时跑 4096 个机器狗时，一个 PPO 迭代可以收集 24 × 4096 ≈ 9.8 万条经验，1500 个迭代就是 1.5 亿步仿真——这是单环境方案无法企及的。

---

## 2. 读代码前必须懂的 12 个概念

> 只听过名词没关系，这一节的目的不是讲透 RL，而是**让你能看懂代码里那些变量名在干什么**。每个概念我都标注了"对应本仓库的哪个变量"。

### 2.1 强化学习的最小循环（MDP）

把"教狗走路"抽象成一个循环：

```
        ┌──────────────────────────────────────────┐
        │                                          │
   观测 obs ──► 策略 π ──► 动作 action ──► 环境 ──┴──► 奖励 reward
   (机器人看到的)  (神经网络)   (12个关节目标角度)  (物理仿真)   (好不好)
                                            │
                                            └──► 新观测 obs' + 是否摔倒 done
```

**五个词**：
- **状态 / 观测（state / observation）**：机器人能感知到的信息。本仓库里是 `obs_buf`，A1 上是 235 维向量。
- **动作（action）**：策略的输出。本仓库里是 12 维（每条腿 3 个关节），范围理论上是 `(-∞, +∞)`，实际被 clip 到 ±100。
- **奖励（reward）**：一个标量，越大越好。本仓库里是 `rew_buf`，由十几个奖励项加权求和得到。
- **策略（policy）**：从观测到动作的映射函数，本仓库里就是 `ActorCritic.actor` 那个 MLP。
- **回合（episode）**：从重置到终止的一段时间，A1 最长 20 秒（1000 个控制步）。

### 2.2 为什么用"概率分布"输出动作？（`self.std`）

初学者最容易困惑的一点：Actor 网络输出的是 12 个均值 `μ`，但真正送进仿真的是**从正态分布 N(μ, σ) 里采样出来的值**。

```python
# rsl_rl/modules/actor_critic.py:119
def update_distribution(self, observations):
    mean = self.actor(observations)
    self.distribution = Normal(mean, mean*0. + self.std)   # ★ 均值来自网络，标准差是独立参数
```

- `self.std` 是网络里一个**可学习参数**（初始 1.0，见 `init_noise_std`），它不是网络输出，是训练中学出来的。
- **训练时采样**（`act`），**测试/部署时直接用均值**（`act_inference`）——这就是为什么 [play.py](legged_gym/scripts/play.py) 里的机器人动作比训练时"干净"。
- 为什么要采样？因为 RL 需要在"探索"和"利用"之间平衡：只输出均值永远不试新动作，就学不出更好的策略。`std` 的大小直接反映探索强度——**训练日志里的 `Mean action noise std` 就是它**，这个值从 ~1.0 慢慢降到 0.1~0.3 是正常收敛的标志；如果一直不降，说明策略还没找到确定性解。

### 2.3 On-policy 是什么意思

PPO 是 **on-policy** 算法：**只能用当前策略刚刚收集的数据来更新当前策略**。收集一批 → 更新几次 → 把数据全扔掉 → 用新策略重新收集。

对比 off-policy 的 SAC/TD3：可以把历史数据存进 replay buffer 反复利用，样本效率高，但并行化困难。

本仓库的选择（4096 个并行环境 × on-policy）本质上是**"用海量采样换掉样本复用"**，因为 GPU 并行让采样变得极其廉价。

代码体现：`RolloutStorage` 每次 `self.alg.update()` 结尾都会 `self.storage.clear()`（[rollout_storage.py:185](rsl_rl/rsl_rl/storage/rollout_storage.py#L185)），把 `step` 指针归零，数据虽然还在显存里但会被直接覆盖。

### 2.4 值函数（Value Function）与"优势"

- **值函数 V(s)**：从状态 s 出发，未来能拿到的累计奖励的期望。相当于"我现在这个姿势，前景如何"。
- **动作价值 Q(s,a)**：在 s 下做动作 a 的前景。
- **优势函数 A(s,a) = Q(s,a) - V(s)**：**这个动作比平均水平好多少**。这是 PPO 的核心信号。

为什么用优势而不是直接用回报？因为回报的绝对值没有意义（有的地形天生奖励就低），我们只关心"相对好坏"。代码里：

```python
# rsl_rl/storage/rollout_storage.py:136
self.advantages = self.returns - self.values
self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)  # ★ 标准化
```

**这个标准化必须记住**：它让 PPO 对奖励的绝对尺度不敏感。所以你把奖励乘以 10 倍，训练结果几乎不变——**很多初学者花时间"调奖励数量级"其实是无效劳动，真正要调的是各项奖励之间的相对比例**。

### 2.5 GAE：怎么算优势（`gamma` 和 `lam`）

不用真实 Q 值，用**多步自举（bootstrapping）** 估计：

```python
# rsl_rl/storage/rollout_storage.py:123  compute_returns()
for step in reversed(range(self.num_transitions_per_env)):
    next_is_not_terminal = 1.0 - self.dones[step].float()
    delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
    advantage = delta + next_is_not_terminal * gamma * lam * advantage   # ★ 从后往前递推
    self.returns[step] = advantage + self.values[step]
```

人话版：
- `delta` = 这一步的"即时奖励 + 下一步估值 - 当前估值" = **时序差分误差 TD-error**
- `advantage` = 把未来所有 TD-error 按 `γλ` 衰减累加 = **GAE（广义优势估计）**
- `next_is_not_terminal` 的作用：**状态终止时切断未来**，不让价值从"摔倒之后"泄露回来。这是一个极易写错的地方。

参数直觉：
- `γ = 0.99`（`gamma`）：折扣因子。控制"看得多远"。γ=0.99 时，"10 秒后"的权重是 0.99^500 ≈ 0.0066，几乎为零；**因此它和 episode 长度是绑定的**。如果你把 `episode_length_s` 从 20 改成 5，γ 应该同步调小（这属于工程经验，值得在面试里提）。
- `λ = 0.95`（`lam`）：GAE 的偏差-方差权衡。λ=0 就是单步 TD（方差小偏差大），λ=1 就是蒙特卡洛（无偏但方差大）。

### 2.6 PPO 的裁剪（clip）：训练稳定的关键

PPO 的目标函数（[ppo.py:154-159](rsl_rl/rsl_rl/algorithms/ppo.py#L154-L159)）：

```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

人话版：
- `ratio` = 新策略产生这个动作的概率 / 旧策略产生它的概率
- 目标是让"好的动作概率升高、坏的动作概率降低"
- 但 **`torch.clamp` 把 ratio 限制在 [0.8, 1.2]**（`clip_param=0.2`）：**如果这次更新已经让策略变化太大，就不再给梯度了**
- 这就是"信任域"思想：与其加复杂的约束（TRPO 的 KL 约束），不如直接用裁剪近似，简单且效果好。**这是 PPO 能统治机器人控制领域的原因。**

### 2.7 自适应学习率（KL 调度）

```python
# rsl_rl/algorithms/ppo.py:139-151
if kl_mean > self.desired_kl * 2.0:
    self.learning_rate = max(1e-5, self.learning_rate / 1.5)     # 变太快 → 降学习率
elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
    self.learning_rate = min(1e-2, self.learning_rate * 1.5)     # 变太慢 → 升学习率
```

`desired_kl = 0.01` 是目标：**每次更新让策略分布只移动"一点点"**。这是 rsl_rl 相比标准 PPO 的一个实用改进，也是**训练曲线里 `Loss/learning_rate` 那条线为什么是锯齿状**的原因。

面试常问："你的学习率是多少？"——标准答案不是"1e-3"，而是"**初始 1e-3，开启 adaptive 调度，随 KL 在 1e-5 ~ 1e-2 之间自动调整**"。

### 2.8 熵正则（`entropy_coef`）

`- self.entropy_coef * entropy_batch.mean()`（[ppo.py:171](rsl_rl/rsl_rl/algorithms/ppo.py#L171)）出现在 loss 里，**带负号且我们最小化 loss，所以等价于最大化熵**。

熵 = 分布的混乱程度。最大化熵 = **鼓励策略保持随机性、不要过早收敛到一个死板解**。A1 用 `entropy_coef = 0.01`。太小会过早收敛到"蹦跳式"局部最优，太大会一直抖动学不会。

### 2.9 向量化环境：为什么是 4096 个

- `num_envs = 4096` 意味着**同时有 4096 只机器狗在训练**，各自在不同地形上。
- 所有环境共享同一个策略网络（权重相同），但**各自独立探索**。收集到的 4096 份经验一起做梯度更新。
- 这既是加速手段，也是**方差缩减手段**：单只狗的运气不好会误导学习，4096 只的平均就不会。
- 代价 1：**显存**。每个环境都要存自己的状态张量。
- 代价 2：**策略是"共享"的**，所以你的观测量设计必须让不同环境下的语义一致（这就是为什么观测里用"机身坐标系下的速度"而不是"世界坐标系速度"）。

### 2.10 Sim-to-Real：为什么代码里到处是"噪声"和"随机"

这个仓库的目标是**迁移到真机**。仿真里学出来的策略直接上真机通常会因为三大差异而崩溃：

| 差异 | 仿真 vs 真机 | 本仓库的对策 | 代码位置 |
|---|---|---|---|
| **观测噪声** | 真机 IMU/编码器有噪声，仿真没有 | 给观测加均匀噪声 | `_get_noise_scale_vec` + `noise` 配置 |
| **动力学差异** | 真机的摩擦、质量、电机响应和仿真不同 | 域随机化（摩擦 64 档、机身质量、外力推） | `_process_rigid_shape_props` / `_process_rigid_body_props` / `_push_robots` |
| **电机建模误差** | 仿真的理想 PD 和真机电机差很多 | 用真机数据训一个 LSTM 执行器网络替代 PD | `Anymal._compute_torques` |

核心思想：**与其把仿真做得更准，不如让策略"见过足够多的不确定性"**，这样它对参数不敏感。这叫 **domain randomization**，是本仓库能上真机的关键。

### 2.11 课程学习（Curriculum Learning）

直接让机器狗在乱石堆上学走路 = 学不会（奖励信号太稀疏，全是失败经验）。
本仓库的解法：**先易后难**。

- **地形课程**：`num_rows=10` 个难度等级，机器人走够远了就升级，走不动就降级（`_update_terrain_curriculum`）。
- **命令课程**（可选，A1 默认关）：速度指令范围随表现逐渐扩大（`update_command_curriculum`）。

### 2.12 非对称 Actor-Critic（`num_privileged_obs`）

`cfg.env.num_privileged_obs = None`（默认）。如果设成一个数字，那么：

- **Actor（策略网络）只用带噪声的观测**——因为它要部署到真机上，只能用真机有的信息
- **Critic（价值网络）可以用"特权观测"**——比如真实速度、地形高度图、接触力，训练时仿真里能拿到，部署时不需要

这叫 **asymmetric actor-critic**，能显著加速训练。**代价**：`on_policy_runner.py:57-60` 会自动把 critic 的输入维度改成 `num_privileged_obs`，你要自己保证环境真的返回了这个维度的 `privileged_obs_buf`。这是一个"知道就好，暂时不用碰"的高级选项。

---

## 3. 文件职责速查表

```
legged_gym/
├── legged_gym/
│   ├── __init__.py                       # 只定义两个路径常量：LEGGED_GYM_ROOT_DIR / LEGGED_GYM_ENVS_DIR
│   ├── envs/
│   │   ├── __init__.py                   # ★ 任务注册处：task_registry.register("a1", ...)
│   │   ├── base/
│   │   │   ├── base_config.py            # ★ BaseConfig：递归把嵌套 class 实例化（配置能这么写的原因）
│   │   │   ├── base_task.py              # ★ 仿真底座：acquire_gym / create_viewer / reset / render
│   │   │   ├── legged_robot_config.py    # ★★ 全部超参默认值（环境 + 训练）
│   │   │   └── legged_robot.py           # ★★★ 核心：观测/奖励/终止/控制/地形/域随机化
│   │   ├── a1/a1_config.py               # A1 机器人任务定义（只覆盖差异）
│   │   ├── anymal_c/                     # ANYmal-C：anymal.py 额外实现了执行器网络
│   │   ├── anymal_b/  cassie/            # 其他机器人（本笔记不展开）
│   ├── scripts/
│   │   ├── train.py                      # ★ 训练入口（只有 3 行有效代码）
│   │   └── play.py                       # ★ 回放/评估/导出 JIT 策略
│   ├── utils/
│   │   ├── __init__.py                   # 统一导出，让 `from legged_gym.utils import *` 生效
│   │   ├── task_registry.py              # ★★ 任务注册表：make_env / make_alg_runner
│   │   ├── helpers.py                    # ★ 命令行参数、配置↔字典转换、checkpoint 路径、JIT 导出
│   │   ├── terrain.py                    # ★ 程序化地形生成（课程地形）
│   │   ├── math.py                       # 三个小工具：yaw 旋转、角度归一化、开方分布随机
│   │   └── logger.py                     # play.py 用的状态记录 + matplotlib 画图
│   └── tests/test_env.py                 # 零动作冒烟测试（验证环境能跑通）
├── rsl_rl/                               # PPO 算法（独立 pip 包，v1.0.2）
│   └── rsl_rl/
│       ├── env/vec_env.py                # ★ 唯一的接口契约（抽象类）
│       ├── modules/actor_critic.py       # ★ Actor / Critic MLP + 高斯策略
│       ├── modules/actor_critic_recurrent.py  # LSTM 版本（可选）
│       ├── algorithms/ppo.py             # ★★ PPO 损失与更新
│       ├── storage/rollout_storage.py    # ★★ 经验缓冲区 + GAE + mini-batch 生成器
│       ├── runners/on_policy_runner.py   # ★★ 训练主循环（唯一真正的"训练循环"在这里）
│       └── utils/utils.py                # RNN 轨迹切分填充工具
├── resources/robots/<name>/{urdf,meshes}/ # URDF + 网格模型（A1 用 .dae）
├── resources/actuator_nets/              # ANYmal 的执行器网络（.pt）
└── logs/<experiment_name>/<date>_<run>/  # 训练输出：model_*.pt + tfevents + exported/policies/
```

### 只标记了星号的 6 个文件（按重要性排序）

| 排序 | 文件 | 一句话理由 | 行数 |
|---|---|---|---|
| 1 | [legged_gym/envs/base/legged_robot.py](legged_gym/envs/base/legged_robot.py) | 任务的全部逻辑 | 906 |
| 2 | [rsl_rl/rsl_rl/runners/on_policy_runner.py](rsl_rl/rsl_rl/runners/on_policy_runner.py) | 训练主循环 | 233 |
| 3 | [legged_gym/envs/base/legged_robot_config.py](legged_gym/envs/base/legged_robot_config.py) | 所有可调参数 | 243 |
| 4 | [rsl_rl/rsl_rl/algorithms/ppo.py](rsl_rl/rsl_rl/algorithms/ppo.py) | 算法本体 | 187 |
| 5 | [legged_gym/utils/terrain.py](legged_gym/utils/terrain.py) | 地形生成 | 187 |
| 6 | [legged_gym/utils/task_registry.py](legged_gym/utils/task_registry.py) | 把上面串起来 | 157 |

---

## 4. 一次训练的完整时间线

先看**入口有多短**——这是理解整个仓库结构的钥匙：

```python
# legged_gym/scripts/train.py:40-43  （全文有效代码就这 4 行）
def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)                    # 造环境
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)  # 造算法
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)                                        # 开训
```

**记住这三个动词**：`make_env` → `make_alg_runner` → `learn`。后面所有内容都是展开这三个调用。

> 注：`from legged_gym.envs import *` 这一行（train.py:36）看似无用，实际上是**必须的副作用导入**——它触发 `envs/__init__.py` 执行所有 `task_registry.register(...)`。删掉它，`--task=a1` 就会报 "Task with name: a1 was not registered"。这是 Python 项目里典型"为了副作用而 import"的用法。

### 4.1 阶段一：`make_env` —— 造一个能跑的物理世界

```
train.py: get_args()                    # 解析命令行（helpers.py:152）
   ↓
task_registry.make_env(name="a1")       # task_registry.py:65
   ├─ get_cfgs("a1")                    # 取出 A1RoughCfg + A1RoughCfgPPO 两个配置对象
   ├─ update_cfg_from_args(...)         # ★ 命令行参数覆盖配置文件（--num_envs / --seed ...）
   ├─ set_seed(env_cfg.seed)            # 固定随机种子（可复现）
   ├─ parse_sim_params(args, sim_params)# 把 cfg.sim 转成 gymapi.SimParams
   └─ LeggedRobot(cfg, sim_params, ...) # ★★ 进入环境构造函数
          ├─ _parse_cfg(cfg)            # 算出 dt / max_episode_length
          └─ BaseTask.__init__          # base_task.py:40
                ├─ gymapi.acquire_gym()                    # 拿到 Isaac Gym 全局实例
                ├─ 分配 obs_buf / rew_buf / reset_buf ...   # ★ 这些张量会一直被复用
                ├─ self.create_sim()      # ★ 子类实现（legged_robot.py:228）
                │     ├─ gym.create_sim(...)                # 创建物理引擎实例
                │     ├─ Terrain(cfg.terrain, num_envs)     # ★ 程序化生成地形网格
                │     ├─ _create_trimesh() / _create_heightfield() / _create_ground_plane()
                │     └─ _create_envs()                     # ★ 循环 4096 次创建环境+机器人
                │           ├─ load_asset(urdf)             # 解析 URDF
                │           ├─ _process_rigid_shape_props() # ★ 摩擦随机化
                │           ├─ _process_dof_props()         # ★ 读出关节限幅、算软限幅
                │           ├─ _process_rigid_body_props()  # ★ 质量随机化
                │           └─ 用名字子串匹配出 feet / 惩罚碰撞 / 终止碰撞的 body 索引
                ├─ gym.prepare_sim(sim)                     # 提交，开始仿真
                └─ create_viewer(...)                       # 非 headless 时开窗口，绑定 ESC/V 键
          ├─ set_camera(...)            # 相机位置
          ├─ _init_buffers()            # ★★ 把所有仿真张量"接"成 PyTorch 张量
          └─ _prepare_reward_function() # ★★ 建立奖励函数表（字符串反射）
```

### 4.2 阶段二：`make_alg_runner` —— 造大脑

```
task_registry.make_alg_runner(env)
   ├─ get_cfgs("a1") -> train_cfg                          # 取 PPO 配置
   ├─ log_dir = logs/rough_a1/<Sep21_13-29-25_>            # ★ 时间戳命名，见下
   ├─ runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device='cuda:0')
   │     ├─ 判断 critic 输入维度（privileged_obs is None → 用 num_obs=235）
   │     ├─ actor_critic = ActorCritic(235, 235, 12, **policy_cfg)  # ★ 建网络
   │     ├─ alg = PPO(actor_critic, **algorithm_cfg)                # ★ 建算法
   │     ├─ alg.init_storage(4096, 24, [235], [None], [12])         # ★ 建经验池
   │     └─ env.reset()                                             # ★ 让所有机器人先站好
   └─ if resume: runner.load(resume_path)                  # 断点续训
```

**日志目录命名规则**（面试可能问到你的实验管理）：
`logs/<experiment_name>/<月日_时分秒>_<run_name>/`
- A1 的 `experiment_name = 'rough_a1'`、`run_name = ''`
- 所以你的那次训练落在 `logs/rough_a1/Sep21_13-29-25_/`
- **一次训练一个时间戳目录**，互不覆盖。想区分实验就改 `run_name`（命令行 `--run_name=lr5e4`）

### 4.3 阶段三：`learn` —— 真正的训练循环

这是**整个仓库的"心脏"**，在 [on_policy_runner.py:102-142](rsl_rl/rsl_rl/runners/on_policy_runner.py#L102-L142)。用伪代码写出来：

```python
for it in range(max_iterations):                    # 外层：1500 次策略更新
    # ---------- 阶段 A：采样（24 步 × 4096 环境）----------
    with torch.inference_mode():                    # 关掉梯度，省显存、加速
        for i in range(num_steps_per_env):          # 24 步
            actions = alg.act(obs, critic_obs)      # ① 策略网络前向 → 采样动作
            obs, _, rewards, dones, infos = env.step(actions)   # ② 仿真推进（见 4.4）
            alg.process_env_step(rewards, dones, infos)         # ③ 存进经验池
        alg.compute_returns(critic_obs)             # ④ GAE 反向递推算优势

    # ---------- 阶段 B：学习（5 epoch × 4 mini-batch = 20 次梯度更新）----------
    value_loss, surrogate_loss = alg.update()       # ⑤ PPO 更新（见 5.5.2）

    if it % save_interval == 0:
        save(f'model_{it}.pt')                      # 每 50 次迭代存一次
```

**关键数字（A1）**：

| 量 | 值 | 说明 |
|---|---|---|
| 每次迭代采样步数 | 24 | `num_steps_per_env` |
| 并行环境数 | 4096 | `num_envs` |
| 每次迭代样本量 | 98,304 | 24 × 4096 |
| 总迭代次数 | 1500 | `max_iterations` |
| **总仿真步数** | **≈ 1.47 亿** | 1500 × 98304 |
| 每次迭代梯度更新次数 | 20 | `num_learning_epochs(5) × num_mini_batches(4)` |
| mini-batch 大小 | 24,576 | 98304 / 4 |

> **为什么 `num_steps_per_env = 24`？** 因为控制频率 50 Hz，24 步 = 0.48 秒 ≈ 机器人走 1~2 个步态周期的长度。太短则 GAE 的自举偏差大，太长则与旧策略差距过大导致 PPO 裁剪失效。这是经验参数，多数四足任务在 20~32 之间。

### 4.4 展开 `env.step()`：一个控制周期里发生了什么

```python
# legged_robot.py:79-103
def step(self, actions):
    self.actions = torch.clip(actions, -100, 100)          # ① 限幅
    self.render()
    for _ in range(self.cfg.control.decimation):           # ② ★ 物理子步循环（decimation=4）
        self.torques = self._compute_torques(self.actions)  #    PD 控制 → 力矩
        self.gym.set_dof_actuation_force_tensor(...)        #    写入力矩
        self.gym.simulate(self.sim)                         #    推进物理 0.005s
        self.gym.refresh_dof_state_tensor(self.sim)         #    刷新关节状态
    self.post_physics_step()                               # ③ 七件事（见下）
    return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
```

**为什么要有 decimation？** 这是本仓库最重要的时间尺度概念：

```
物理仿真（Isaac Gym）：  dt = 0.005 s   → 200 Hz   ← 保证接触求解稳定性
     ↓ 连续算 4 次同一个动作
策略推理（神经网络）：    dt = 0.02  s   →  50 Hz   ← 保证网络来得及算
```

`decimation = 4`（[a1_config.py:47](legged_gym/envs/a1/a1_config.py#L47)）意味着：**神经网络每 0.02 秒出一次动作，这 4 个物理步里力矩都保持不变**。

- `self.dt = decimation × sim_params.dt = 0.02`（[legged_robot.py:729](legged_gym/envs/base/legged_robot.py#L729)）
- **所有奖励函数的"每秒"概念、episode 长度、推外力间隔，都是基于 `self.dt` 算的。**
- 真机的关节伺服通常也是 500~1000 Hz 内环 + 50 Hz 策略，这套结构和真机是对得上的。

### 4.5 展开 `post_physics_step()`：每步必做的 7 件事

```python
# legged_robot.py:105-136   —— 顺序很重要！
def post_physics_step(self):
    self.gym.refresh_actor_root_state_tensor(self.sim)    # ⓪ 从 GPU 取回仿真状态
    self.gym.refresh_net_contact_force_tensor(self.sim)

    self.episode_length_buf += 1                          # ① 计时
    self.common_step_counter += 1

    self.base_quat[:] = self.root_states[:, 3:7]          # ② 把世界系状态转到机体系
    self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
    self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
    self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

    self._post_physics_step_callback()                    # ③ 重采样命令 / 测地形高度 / 推机器人

    self.check_termination()                              # ④ 判定摔倒
    self.compute_reward()                                 # ⑤ 算奖励
    env_ids = self.reset_buf.nonzero().flatten()
    self.reset_idx(env_ids)                               # ⑥ 重置摔了的机器人
    self.compute_observations()                           # ⑦ 算新观测

    self.last_actions[:] = self.actions[:]                # ⑧ 记录历史（供 action_rate 奖励用）
    self.last_dof_vel[:] = self.dof_vel[:]
    self.last_root_vel[:] = self.root_states[:, 7:13]
```

**② 这一步是"坐标系"的关键**，初学者最容易忽略：

`root_states` 是**世界坐标系**下的状态，但神经网络绝对不能直接吃世界系速度——因为"世界系的 +1 m/s"意味着"永远朝东北走"，而这个任务要的是"朝自己面向的前方走"。所以必须用 `quat_rotate_inverse` 转成**机身坐标系**。

- `base_lin_vel`：机体系线速度，**这才是"前进速度"**
- `projected_gravity`：把世界系的 (0,0,-1) 重力方向转到机体系。**平坦站立时 = [0, 0, -1]；倾斜时前两个分量非零**。所有"保持平衡"的奖励都基于它，它是一个优雅的"姿态传感器"。

**⑦ 的顺序陷阱**：`compute_observations` 在 `reset_idx` 之后调用，而 `env.step` 里最后才更新 `last_actions`。所以观测里的 `self.actions` 段是**当前动作**（不是上一步动作），`last_actions` 只用于算 `action_rate` 奖励。这是上游代码一个容易让人读错的地方。

---

## 5. 核心文件精读

### 5.1 legged_robot.py（最重要的文件）

#### 5.1.1 关键行号速查（先收藏这张表）

| 功能 | 方法 | 行号 |
|---|---|---|
| **主循环：物理步进 + 控制** | `step` | [79-103](legged_gym/envs/base/legged_robot.py#L79-L103) |
| **每步 7 件事** | `post_physics_step` | [105-136](legged_gym/envs/base/legged_robot.py#L105-L136) |
| **终止判定** | `check_termination` | [138-143](legged_gym/envs/base/legged_robot.py#L138-L143) |
| **重置** | `reset_idx` | [145-188](legged_gym/envs/base/legged_robot.py#L145-L188) |
| **奖励求和** | `compute_reward` | [190-207](legged_gym/envs/base/legged_robot.py#L190-L207) |
| ★ **观测构造** | `compute_observations` | [209-226](legged_gym/envs/base/legged_robot.py#L209-L226) |
| **创建仿真与地形** | `create_sim` | [228-244](legged_gym/envs/base/legged_robot.py#L228-L244) |
| ★ **摩擦随机化** | `_process_rigid_shape_props` | [254-277](legged_gym/envs/base/legged_robot.py#L254-L277) |
| **关节限幅** | `_process_dof_props` | [279-305](legged_gym/envs/base/legged_robot.py#L279-L305) |
| **质量随机化** | `_process_rigid_body_props` | [307-318](legged_gym/envs/base/legged_robot.py#L307-L318) |
| **每步回调（命令/高度/推）** | `_post_physics_step_callback` | [320-335](legged_gym/envs/base/legged_robot.py#L320-L335) |
| **重采样速度指令** | `_resample_commands` | [337-351](legged_gym/envs/base/legged_robot.py#L337-L351) |
| ★★ **力矩计算（PD）** | `_compute_torques` | [353-375](legged_gym/envs/base/legged_robot.py#L353-L375) |
| **重置关节** | `_reset_dofs` | [377-391](legged_gym/envs/base/legged_robot.py#L377-L391) |
| **重置机身位姿** | `_reset_root_states` | [392-412](legged_gym/envs/base/legged_robot.py#L392-L412) |
| **推机器人** | `_push_robots` | [414-419](legged_gym/envs/base/legged_robot.py#L414-L419) |
| ★ **地形课程** | `_update_terrain_curriculum` | [421-441](legged_gym/envs/base/legged_robot.py#L421-L441) |
| **噪声向量** | `_get_noise_scale_vec` | [455-478](legged_gym/envs/base/legged_robot.py#L455-L478) |
| ★ **张量初始化** | `_init_buffers` | [481-542](legged_gym/envs/base/legged_robot.py#L481-L542) |
| ★★ **奖励函数表** | `_prepare_reward_function` | [544-567](legged_gym/envs/base/legged_robot.py#L544-L567) |
| ★ **创建环境 + 名字匹配** | `_create_envs` | [614-699](legged_gym/envs/base/legged_robot.py#L614-L699) |
| **环境初始位置** | `_get_env_origins` | [701-726](legged_gym/envs/base/legged_robot.py#L701-L726) |
| ★ **时间尺度推导** | `_parse_cfg` | [728-738](legged_gym/envs/base/legged_robot.py#L728-L738) |
| **高度采样点** | `_init_height_points` / `_get_heights` | [761-813](legged_gym/envs/base/legged_robot.py#L761-L813) |
| **17 个奖励函数** | `_reward_*` | [816-906](legged_gym/envs/base/legged_robot.py#L816-L906) |

#### 5.1.2 `_parse_cfg`：所有时间尺度的源头（[728-738](legged_gym/envs/base/legged_robot.py#L728-L738)）

```python
def _parse_cfg(self, cfg):
    self.dt = self.cfg.control.decimation * self.sim_params.dt      # 0.005 × 4 = 0.02 s ★
    self.obs_scales = self.cfg.normalization.obs_scales             # 观测缩放
    self.reward_scales = class_to_dict(self.cfg.rewards.scales)     # ★ 奖励配置 → 普通 dict
    self.command_ranges = class_to_dict(self.cfg.commands.ranges)
    if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
        self.cfg.terrain.curriculum = False                         # 平地不支持地形课程
    self.max_episode_length_s = self.cfg.env.episode_length_s
    self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)   # 20/0.02 = 1000 步
    self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)  # 15/0.02 = 750 步
```

**这一共 7 行，但是所有"秒"和"步"的换算都在这。** 如果你改了 `sim.dt` 或 `decimation`，所有和步数相关的量都会跟着变——包括你以为固定的 "20 秒 episode" 和 "每 15 秒推一次"。

`class_to_dict`（[helpers.py:41](legged_gym/utils/helpers.py#L41)）的作用：把 `cfg.rewards.scales` 这个嵌套的配置类变成普通字典，这样才能 `.items()` 遍历、才能 `pop` 掉零项。

#### 5.1.3 `_compute_torques`：一行公式决定机器人的"肌肉"（[353-375](legged_gym/envs/base/legged_robot.py#L353-L375)）

**这是整个仓库最核心的一行代码：**

```python
# legged_robot.py:365-368
actions_scaled = actions * self.cfg.control.action_scale
if control_type == "P":
    torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

写成数学公式（对每个关节）：

$$
\tau_i = K_p^{(i)} \cdot \underbrace{\left( \alpha \cdot a_i + q_i^{default} - q_i \right)}_{e_i\ :\ 位置误差} - K_d^{(i)} \cdot \dot q_i
$$

**逐项拆解**：

| 符号 | 代码 | A1 的值 | 含义 |
|---|---|---|---|
| $a_i$ | `actions` | 网络输出，±100 被 clip | 策略输出 |
| $\alpha$ | `action_scale` | **0.25** | 一个动作单位对应 0.25 rad 的目标偏移 |
| $q_i^{default}$ | `default_dof_pos` | 见 `default_joint_angles` | 网络输出全 0 时的站立姿态 |
| $q_i$ | `dof_pos` | 仿真当前角度 | 实际关节角 |
| $K_p$ | `stiffness` | **20**（A1 所有关节） | 位置刚度 |
| $K_d$ | `damping` | **0.5** | 速度阻尼 |
| $\tau_{limit}$ | 从 URDF 读的 `effort` | 髋 20 / 大腿 55 / 小腿 55 | 力矩限幅 |

**三个必须记住的设计含义**：

1. **策略输出的是"关节目标角的偏移量"，不是力矩。** 这就是为什么 A1 的 `action_scale = 0.25`：网络输出 ±1 对应 ±0.25 rad ≈ ±14° 的位置调整。**策略学的其实是"脚该往哪摆"，而 PD 控制器负责"怎么用力去够到那个位置"**——低层控制稳定的任务交给了 PD，网络只需要学高层决策。这就是为什么这个框架能让机器狗在几十分钟内学会走路。

2. **`default_dof_pos` 相当于动作的"零点"**。所以 `default_joint_angles` 必须是一个**合理的站立姿态**。A1 的值（[a1_config.py:39-65](legged_gym/envs/a1/a1_config.py#L39-L64)是：

   | 关节 | 值 | 理由 |
   |---|---|---|
   | `*_hip_joint` | ±0.1 | 左腿 +0.1，右腿 −0.1，让腿略微外张 |
   | `*_thigh_joint` | 0.8（前腿）/ 1.0（后腿） | 大腿前倾 |
   | `*_calf_joint` | **−1.5** | 小腿后收，与大腿构成"<"形（膝盖前弓） |

   如果这个姿态设错了（比如小腿设成 0），机器人一开始就是"腿伸直"的，PD 会把所有关节拼命往一个不合理的位置拉，训练会非常困难。**这是迁移新机器人时最常出错的地方之一。**

3. **力矩一定要 clip。** 因为 PD 在"误差巨大"时算出的力矩是荒唐的（比如重置瞬间的位置误差可能让力矩到几万牛米）。`torque_limits` 来自 URDF 的 `effort` 字段（[legged_robot.py:299](legged_gym/envs/base/legged_robot.py#L299)），A1 是髋 20 N·m、大腿/小腿 55 N·m。**真机上电机也是有这个上限的，所以这个 clip 同时也是"仿真→真机"的一致性问题。**

**`control_type` 的三种模式**（[legged_robot.py:367-374](legged_gym/envs/base/legged_robot.py#L367-L374)）：

- `'P'`：位置控制（**默认，四足标准做法**），策略输出位置目标
- `'V'`：速度控制，用位置误差的**差分**近似加速度项
- `'T'`：直接力矩控制，`torques = actions * action_scale`。**最难训**，但有些人形机器人任务会用（因为人形的接触相位更复杂，位置控制不够灵活）

#### 5.1.4 `compute_observations`：机器人"看到"什么（[209-226](legged_gym/envs/base/legged_robot.py#L209-L226)）

```python
self.obs_buf = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,                          # [0:3]
    self.base_ang_vel * self.obs_scales.ang_vel,                          # [3:6]
    self.projected_gravity,                                               # [6:9]
    self.commands[:, :3] * self.commands_scale,                           # [9:12]
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,      # [12:24]
    self.dof_vel * self.obs_scales.dof_vel,                               # [24:36]
    self.actions                                                          # [36:48]
), dim=-1)
if self.cfg.terrain.measure_heights:
    heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
    self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)             # [48:235]
if self.add_noise:
    self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec   # ★ 加噪
```

**A1 的 235 维逐个拆开**（这是必须能背下来的表）：

| 区间 | 维度 | 内容 | 缩放 | 真机对应传感器 |
|---|---|---|---|---|
| `[0:3]` | 3 | 机体系线速度 `base_lin_vel` | ×2.0 | 状态估计器（仿真直接给真值） |
| `[3:6]` | 3 | 机体系角速度 `base_ang_vel` | ×0.25 | IMU 陀螺仪 |
| `[6:9]` | 3 | 投影重力 `projected_gravity` | ×1.0 | IMU 姿态解算 |
| `[9:12]` | 3 | 指令 (vx, vy, ωz) | ×[2, 2, 0.25] | 上层规划器 / 遥控器 |
| `[12:24]` | **12** | 关节角偏差 `dof_pos − default_dof_pos` | ×1.0 | 关节编码器 |
| `[24:36]` | **12** | 关节角速度 `dof_vel` | ×0.05 | 编码器微分 |
| `[36:48]` | **12** | 上一步动作 `actions` | ×1.0 | 无（内部记忆） |
| `[48:235]` | **187** | 地形高度扫描 | ×5.0 | 激光雷达 / 深度相机 |

**几个为什么**：

- **`num_observations = 235` 是怎么来的？** `3+3+3+3 + 3×12 + 187 = 48 + 187 = 235`。其中 187 来自高度采样网格：`len(measured_points_x)=17` × `len(measured_points_y)=11` = 187（[legged_robot_config.py:54-55](legged_gym/envs/base/legged_robot_config.py#L54-L55)）。
  **！！！这是迁移新机器人时必须重算的数字。迁移到 23 自由度的 G1 时，应该是 12 + 69 + 187 = 268（或关掉高度测量后为 81）。写错了不会报错，但训练必然失败。**

- **为什么要 `− default_dof_pos`？** 让"站立姿态"对应全 0。网络不需要去学一个恒定的偏置，输入天然是零均值的，训练更稳。

- **为什么要 ×缩放系数？** 为了让所有输入量级大致在 ±1 附近。关节速度可能有 20 rad/s，不缩放会淹没其他信号；线速度 ±1.5 m/s 乘 2 变成 ±3。**这是神经网络训练的工程常识，不是可有可无的美化。**

- **187 维高度是怎么测的？**（[`_get_heights`, 777-813](legged_gym/envs/base/legged_robot.py#L777-L813)）**不是射线检测**，而是查表：

  ```python
  points = quat_apply_yaw(base_quat, height_points) + root_states[:, :3]   # 采样点转到世界系（只转 yaw）
  points = (points / horizontal_scale).long()                              # 转成地形栅格索引
  heights1 = height_samples[px, py];  heights2 = height_samples[px+1, py];  heights3 = height_samples[px, py+1]
  heights = min(heights1, heights2, heights3)                              # ★ 取三点最小值（保守）
  ```
  把机器人周围 1.6 m × 1.0 m 的 17×11 个点投影到地形高度图栅格上直接查高度。取 `min` 是为了让策略对地面估计偏保守（宁可认为地面更低）。**这是一个训练出来的工程技巧，如果取均值或线性插值，训练效果会变差。**

- **真实的高度值**是 `z_base − 0.5 − z_terrain`，即"机身相对地面 0.5 m 以下的高度"，clip 到 [−1, 1]。这样机器人能感知到自己"离地多高、脚下是坑还是有坎"。

- **噪声 `noise_scale_vec`**（[`_get_noise_scale_vec`, 455-478](legged_gym/envs/base/legged_robot.py#L455-L478)）的分段必须和上面的 obs 结构**逐段对齐**（比如 `noise_vec[12:24]` 对应 dof_pos）。**如果你改了 obs 结构却忘了改这个函数，噪声就会加到错误的段上。** 注意 `[9:12]`（指令）和 `[36:48]`（动作）被显式设成 0——指令是上层给的精确值，动作是自己算的，都不需要加噪。

#### 5.1.5 奖励机制：`compute_reward` + `_prepare_reward_function`

```python
# 544-567  准备阶段：只做一次
for key in list(self.reward_scales.keys()):
    scale = self.reward_scales[key]
    if scale == 0:
        self.reward_scales.pop(key)          # ★ 零分的奖励直接删掉，函数都不会被调用
    else:
        self.reward_scales[key] *= self.dt   # ★ 非零的乘以 dt
self.reward_functions = [getattr(self, '_reward_' + n) for n in self.reward_scales if n != "termination"]

# 190-207  运行时：每步都做
def compute_reward(self):
    self.rew_buf[:] = 0.
    for i in range(len(self.reward_functions)):
        name = self.reward_names[i]
        rew = self.reward_functions[i]() * self.reward_scales[name]     # ★ 调用 + 加权
        self.rew_buf += rew
        self.episode_sums[name] += rew                                   # ★ 分项累计，用于日志
    if self.cfg.rewards.only_positive_rewards:
        self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)            # ★ 总奖励裁剪到 ≥0
    if "termination" in self.reward_scales:                              # ★ 终止奖励在裁剪之后加
        rew = self._reward_termination() * self.reward_scales["termination"]
        self.rew_buf += rew
```

**四个必须理解的细节**：

1. **零分配的奖励完全不执行**。所以把 `torques = -0.0` 改成 `-0.0002` 不只是"加了个惩罚"，而是**多了一次 GPU 上的张量运算**。调试时把不相关的奖励设成 0 能显著提速。

2. **所有奖励乘以 `dt`**。这样奖励的含义从"每步得分"变成了"**每秒得分率**"。好处：改 `decimation`（控制频率）时，奖励的物理含义不变。日志里 `rew_xxx` 显示的也是"每秒平均得分"（[reset_idx:179](legged_gym/envs/base/legged_robot.py#L179) 除以 `max_episode_length_s`）。

3. **`only_positive_rewards = True`（A1 继承默认值）**：总奖励被裁剪到 ≥ 0。这意味着**惩罚项只能"抵消"奖励，不能制造负奖励**。这个设计配合"摔倒了就重置"机制，避免了早期策略因为不断获得负奖励而陷入"我什么都别做"的退化解。
   ⚠️ **副作用**：如果某个负奖励的绝对值超过了正奖励总和，实际效果就被"削平"了，你调大它的数值不会有效果。**所以调参时要看日志里的分项 `rew_xxx`，而不是只看总数。**

4. **`termination` 奖励在 clip 之后加。** 因为如果要惩罚"摔倒"（通常是很大的负数），放在 clip 前会被削平，放在 clip 后才能真正生效。

**⚠️ 陷阱：A1 的 `termination = -0.0` 等价于 0**（Python 里 `-0.0 == 0` 为 True），所以它被 `pop` 掉了——**A1 实际上没有任何终止惩罚**，摔倒只是重置，不给额外惩罚。你可以验证：`compute_reward` 里 `if "termination" in self.reward_scales` 对 A1 是 False。

#### 5.1.6 A1 的 10 个生效奖励（逐个讲清 + 调参直觉）

先做个统计：`A1RoughCfg.rewards.scales` 里声明了 15 项 + A1 自己加的 1 项，其中 **6 项是 `-0.0` 被删掉**，最终生效的是这 10 项：

| 奖励名 | 配置值 | 公式 | 直觉 | 调参方向 |
|---|---|---|---|---|
| `tracking_lin_vel` | **+1.0** | $\exp(-\|v_{cmd,xy}-v_{xy}\|^2/0.25)$ | **主奖励**：跟上速度指令 | 想跑得更准就加大 |
| `tracking_ang_vel` | **+0.5** | $\exp(-(\omega_z^{cmd}-\omega_z)^2/0.25)$ | 主奖励：跟上转向指令 | 转向差就加大 |
| `lin_vel_z` | **−2.0** | $v_z^2$ | 别上下弹跳 | 机器人蹦跳时加大 |
| `ang_vel_xy` | **−0.05** | $\omega_x^2+\omega_y^2$ | 别翻滚摇摆 | 机身晃动时加大 |
| `torques` | **−0.0002** | $\sum\tau_i^2$ | 省力、省电、减少发热 | 真机迁移时要适当加大 |
| `dof_acc` | **−2.5e-7** | $\sum(\Delta\dot q/\Delta t)^2$ | 别猛抽关节（保护减速器） | 抖动严重时加大 |
| `feet_air_time` | **+1.0** | $(\text{腾空时间}-0.5)\cdot\mathbb{1}_{\text{首次触地}}$ | **鼓励抬腿迈步**，别贴地蹭 | 步态太小碎时加大 |
| `collision` | **−1.0** | 大腿/小腿碰撞次数 | 别用腿撞地 | — |
| `action_rate` | **−0.01** | $\sum(a_t-a_{t-1})^2$ | 动作要平滑 | 输出抖动时加大 |
| `dof_pos_limits` | **−10.0** | 超出软限幅的量 | 别把关节顶死 | 关节经常打到限位时加大 |

**被删掉的 6 项**（值都是 `-0.0`）：`termination`、`orientation`、`dof_vel`、`base_height`、`feet_stumble`、`stand_still`。

> **⚠️ 重要发现：`base_height` 和 `orientation` 在 A1 上是关的！** 这意味着 A1 的"保持趴平"和"保持机身高度"**完全靠 `ang_vel_xy` 和 `tracking_lin_vel` 间接约束**。如果你想让 A1 站得更稳，第一件事可以是把 `orientation = -0.5` 和 `base_height = -1.0` 打开——但要注意 `base_height_target` 也要设对（A1 配置里是 0.25）。

**几个奖励函数的代码细节**：

<details>
<summary><b>feet_air_time（最巧妙的一个）</b> —— 点击展开</summary>

```python
# legged_robot.py:882-893
def _reward_feet_air_time(self):
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.        # 足端 z 向接触力 > 1N
    contact_filt = torch.logical_or(contact, self.last_contacts)       # ★ 用了"或"，见下
    self.last_contacts = contact
    first_contact = (self.feet_air_time > 0.) * contact_filt           # 腾空后第一次着地
    self.feet_air_time += self.dt                                      # 累计腾空时间
    rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
    rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1       # ★ 静止时不给奖励
    self.feet_air_time *= ~contact_filt                                # 着地后清零
    return rew_airTime
```

**为什么要 `contact_filt = contact OR last_contacts`？** 注释里写了原因："PhysX 在网格地面上的接触报告不可靠"——足端接触在三角网格上会**闪烁**（这一步有接触、下一步没有）。用"或"相当于一个滤波器：**只要最近两步中有一步接触过，就认为还在接触**。这个细节如果去掉，`feet_air_time` 会算得一塌糊涂。

**`(air_time - 0.5)` 的含义**：腾空 0.5 秒以下反而扣分，0.5 秒以上才加分。这样鼓励大步幅而不是小碎步。

**最后一行 `*= (norm(commands) > 0.1)`**：指令接近静止时不给奖励。否则机器人会为了骗奖励原地抬腿。
</details>

<details>
<summary><b>tracking_lin_vel：为什么用 exp 而不是直接误差</b> —— 点击展开</summary>

```python
# legged_robot.py:872-875
def _reward_tracking_lin_vel(self):
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)   # tracking_sigma = 0.25
```

$\exp(-e^2/\sigma)$ 的形状：
- 误差 e=0 时 → 1.0（满分）
- 误差 e=0.5 m/s 时 → $e^{-0.25/0.25} = 0.37$（还有 37% 分）
- 误差 e=1.5 m/s 时 → $e^{-2.25/0.25} = 0.00012$（几乎 0）

**好处 1**：有界在 [0,1]，不会像 $-\|e\|^2$ 那样在速度误差大时产生巨大的负梯度把网络带崩。
**好处 2**：`tracking_sigma` 控制"宽容度"——σ 越大越宽容。**这是调参时的关键旋钮**：如果机器人老是慢慢走不敢加速，把 σ 调小（比如 0.1）逼它跟上；如果训练初期学不动，调大（比如 0.5）。
**好处 3**：这个奖励的上界固定为 1.0，所以 `update_command_curriculum` 可以用"是否达到 80% 满分"作为升级判据（[legged_robot.py:450](legged_gym/envs/base/legged_robot.py#L450)）。
</details>

<details>
<summary><b>dof_pos_limits：软限幅怎么来的</b> —— 点击展开</summary>

关节限幅不是直接用 URDF 的物理限幅，而是在 `_process_dof_props`（[295-304](legged_gym/envs/base/legged_robot.py#L295-L304)）里被"收紧"了：

```python
m = (lower + upper) / 2          # 中点
r = upper - lower                # 全行程
lower_new = m - 0.5 * r * soft_dof_pos_limit   # soft_dof_pos_limit = 0.9 (A1)
upper_new = m + 0.5 * r * soft_dof_pos_limit
```

A1 的 `soft_dof_pos_limit = 0.9` → 惩罚区间缩到原行程的 90%，**留 5% 的安全余量**。这样惩罚会在**碰到物理限位之前**就生效，避免仿真里关节撞死。

```python
def _reward_dof_pos_limits(self):
    out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)   # 低于下限的部分
    out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)   # 高于上限的部分
    return torch.sum(out_of_limits, dim=1)
```
`clip(max=0.)` 使得"在范围内时为 0"，只有越界才有正值。**这是 RL 奖励里"单边惩罚"的标准写法**，`dof_vel_limits` / `torque_limits` 用的是同一套路。
</details>

#### 5.1.7 ⚠️ **上游隐藏 Bug**：`feet_stumble` 名字对不上

我在本仓库里核实了配置声明与代码实现的一一对应关系，发现**唯一一处不匹配**：

```
配置里写的  : feet_stumble = -0.0            (legged_robot_config.py:144)
代码里实现的: def _reward_stumble(self)      (legged_robot.py:895)
```

`_prepare_reward_function` 会去找 `_reward_feet_stumble`（[563](legged_gym/envs/base/legged_robot.py#L563)），**这个方法不存在**。之所以至今没报错，纯粹因为它是 `-0.0`，被 `pop` 掉了。

**验证方法**：把配置改成 `feet_stumble = -0.1`（非零），训练立刻崩：

```
AttributeError: 'LeggedRobot' object has no attribute '_reward_feet_stumble'
```

**这件事的价值**：
1. 你自己踩过这个坑，就知道"**奖励名必须和 `_reward_` 后面的方法名严格一致**"。加奖励时先 grep 确认。
2. **这是一个很好的面试话题**——它说明你不是"跑通了就跑通了"，而是真的读了代码、核对了实现与配置的对应关系。可以这样讲："我在读代码时发现上游有个潜在 bug，配置项 `feet_stumble` 对应的 `_reward_feet_stumble` 方法根本不存在，只是因为该项默认是 0 才没暴露。我用脚本核对了两边名字的一一对应，确认只有这一处。" 这比背十篇论文摘要都更有说服力。

> 想修的话：把 `legged_robot.py:895` 的方法改名为 `_reward_feet_stumble` 即可。

#### 5.1.8 终止与重置：什么时候"摔倒"

```python
# legged_robot.py:138-143
def check_termination(self):
    self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
    self.time_out_buf = self.episode_length_buf > self.max_episode_length   # ★ 超时
    self.reset_buf |= self.time_out_buf
```

**只有两种终止条件**：
1. **非法接触**：`terminate_after_contacts_on` 里的 body（A1 是 `base`，即机身触地）受到 > 1N 的力 → 摔倒
2. **超时**：跑满 1000 步（20 秒）

`termination_contact_indices` 是**名字子串匹配**出来的（[657-659](legged_gym/envs/base/legged_robot.py#L657-L659) + [697-699](legged_gym/envs/base/legged_robot.py#L697-L699)）：

```python
termination_contact_names = [s for name in cfg.asset.terminate_after_contacts_on for s in body_names if name in s]
```

**注意是子串包含，不是精确相等！** A1 配 `terminate_after_contacts_on = ["base"]`，会匹配到所有名字含 "base" 的 body。**迁移新机器人时的坑**：如果 URDF 里同时有 `base_link` 和 `base_footprint`，你会意外地把 `base_footprint`（通常是个应该贴地的虚拟 link）也算进终止条件，机器人会一碰地就"摔倒"。

**`time_out_buf` 的妙用**（[rsl_rl/algorithms/ppo.py:107-109](rsl_rl/rsl_rl/algorithms/ppo.py#L107-L109)）：

```python
if 'time_outs' in infos:
    self.transition.rewards += self.gamma * self.transition.values * infos['time_outs']
```

**超时不是"失败"，不应该像摔倒那样切断价值**。所以代码在超时的那一步把"未来的价值估计"补回给奖励——这叫 **bootstrapping on timeouts**。如果不做这件事，机器人学会的是"在 19.9 秒时故意摔倒"，因为反正超时的回报是 0 而摔倒也是 0。**这是一个初学者几乎一定会忽略、但在面试里能显著加分的技术细节。**

`_reset_dofs` 与 `_reset_root_states`（[377-412](legged_gym/envs/base/legged_robot.py#L377-L412)）——重置**不是复位到同一状态**，而是**随机化**：

```python
self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, ...)     # 关节角在默认值的 0.5~1.5 倍
self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, ...)                 # 初始速度 ±0.5
self.root_states[env_ids, :2] += torch_rand_float(-1., 1., ...)                    # xy 位置 ±1m
```

**为什么要随机化初始状态？** 让策略见到更多起始条件，鲁棒性更强。如果每次都从同一个精确姿态开始，策略会过拟合到这个初始状态（真机部署时根本站不起来——这是经典的 sim2real 失败模式）。

#### 5.1.9 域随机化：三个入口

| 随机化什么 | 在哪个函数 | 什么时候执行 | A1 的配置 |
|---|---|---|---|
| **摩擦系数** | `_process_rigid_shape_props` [:254](legged_gym/envs/base/legged_robot.py#L254) | **建环境时**（每环境一次） | `randomize_friction=True`，[0.5, 1.25]，**64 档分桶** |
| **机身质量** | `_process_rigid_body_props` [:307](legged_gym/envs/base/legged_robot.py#L307) | **建环境时** | A1 默认 **False**！（ANYmal-C 是 ±5 kg） |
| **外力推搡** | `_push_robots` [:414](legged_gym/envs/base/legged_robot.py#L414) | **训练中每 750 步**（15 秒） | `push_robots=True`，±1.0 m/s |

**摩擦的"64 档分桶"是个性能技巧**（[267-273](legged_gym/envs/base/legged_robot.py#L267-L273)）：

```python
if env_id == 0:                                       # ★ 只在第一个环境创建时算一次
    num_buckets = 64
    bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
    friction_buckets = torch_rand_float(range[0], range[1], (num_buckets, 1), device='cpu')
    self.friction_coeffs = friction_buckets[bucket_ids]     # 4096 个环境只用了 64 个不同的摩擦值
```

**为什么不用 4096 个不同的随机值？** 因为 PhysX 底层要为每个不同的摩擦值组合生成接触材质，值越离散越省内存。**这是"用一点点多样性换取大大降低显存"的工程折中**。理解这一点，你就理解了为什么大规模并行仿真里到处是这种"分桶"技巧。

**推搡用的是"直接改速度"而不是"施加力"**（[417-418](legged_gym/envs/base/legged_robot.py#L417-L418)）：

```python
self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2))   # 直接改线速度
self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
```
注释写了："Emulates an impulse by setting a randomized base velocity"——**用瞬时速度突变来模拟冲量**，比真的施加力更简单可控（因为物理引擎里的力是持续作用的，而冲量是瞬时的）。

#### 5.1.10 地形课程：`_update_terrain_curriculum`（[421-441](legged_gym/envs/base/legged_robot.py#L421-L441)）

```python
distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)   # 走了多远
move_up   = distance > self.terrain.env_length / 2                                            # > 4 m 就升级
move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down                                    # ★ 升级/降级
self.terrain_levels[env_ids] = torch.where(self.terrain_levels >= self.max_terrain_level,      # 满级重置到随机难度
                                           torch.randint_like(self.terrain_levels, self.max_terrain_level), ...)
self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]  # ★ 换出生点
```

**判据的物理含义**：
- **升级**：一个 episode（最多 20 秒）里走出了 4 m（= 地形块长度的一半）
- **降级**：走了不到"指令速度 × 10 秒"的距离——**note 这里的 10 秒 = `max_episode_length_s × 0.5`**，意思是"如果只发挥了指令速度的一半都不到，说明这个难度太高了"
- **满级**：回到随机难度（保持探索，不让所有环境都挤在最高级）

**这一机制的实现手法很值得学**：`env_origins` 是每个环境的出生点坐标。切换难度 = **直接改出生点坐标**。因为课程地形是 10×20 的网格，每个格子是一块地形，所以"换难度"就只是换个格子坐标。**没有重建任何东西，开销几乎为零。** 这种"用数据表示状态、避免重建资源"的思路在大规模仿真里随处可见。

#### 5.1.11 `_get_env_origins`：出生点怎么定（[701-726](legged_gym/envs/base/legged_robot.py#L701-L726)）

```python
if mesh_type in ["heightfield", "trimesh"]:              # 有地形
    self.custom_origins = True
    self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,))     # 初始难度 0~5
    self.terrain_types  = arange(num_envs) // (num_envs / num_cols)             # ★ 均匀分配到 20 种地形
    self.env_origins[:] = self.terrain_origins[terrain_levels, terrain_types]
else:                                                     # 平地：摆网格
    env_origins[:, 0] = spacing * xx.flatten();  env_origins[:, 1] = spacing * yy...
```

`terrain_types` 那一行保证了 **4096 个环境均匀铺在 20 种地形类型上**（每种 205 个左右），而不是随机分配（随机分配会有统计涨落）。同时 `terrain_levels` 是随机的 → **同一个难度等级上，各种地形类型都在被训练**。这是课程学习和多样性之间的平衡设计。

### 5.2 legged_robot_config.py 参数总表

**先记这张"最常改的 10 个参数"表**（按你实际会碰到的频率排序）：

| 参数 | A1 值 | 改了会怎样 | 位置 |
|---|---|---|---|
| `env.num_envs` | 4096 | 显存不够就调小（1024/2048）；调小会变慢、方差变大 | [cfg:35](legged_gym/envs/base/legged_robot_config.py#L35) |
| `env.num_observations` | 235 | **改机器人必须重算**，见 8.4 节 | [cfg:36](legged_gym/envs/base/legged_robot_config.py#L36) |
| `env.num_actions` | 12 | = 关节数 | [cfg:38](legged_gym/envs/base/legged_robot_config.py#L38) |
| `env.episode_length_s` | 20 | 太短则学不到长程步态，太长则浪费算力 | [cfg:41](legged_gym/envs/base/legged_robot_config.py#L41) |
| `control.action_scale` | 0.25 | 策略的"动作分辨率"；太小则腿摆不开，太大则抖动 | [a1:47](legged_gym/envs/a1/a1_config.py#L46) |
| `control.stiffness/damping` | 20 / 0.5 | **最影响"能不能站住"的两个数**，见 8.3 节 | [a1:44-45](legged_gym/envs/a1/a1_config.py#L43-L45) |
| `rewards.scales.*` | 见 5.1.6 | 决定行为，改一个要观察一整轮训练 | [a1:74-78](legged_gym/envs/a1/a1_config.py#L73-L77) |
| `domain_rand.*` | 见 5.1.9 | 从"仿真跑得好看"到"真机能用"的关键 | [cfg:121-128](legged_gym/envs/base/legged_robot_config.py#L121-L128) |
| `terrain.num_rows/num_cols` | 10 / 20 | 课程等级数与地形类型数 | [cfg:61-62](legged_gym/envs/base/legged_robot_config.py#L61-L62) |
| `runner.max_iterations` | 1500 | 训练总时长；ANYmal 平地版只要 300 | [cfg:234](legged_gym/envs/base/legged_robot_config.py#L234) |

**分组速查**（完整版）：

<details>
<summary><b>① env 环境基本设置</b></summary>

| 参数 | 默认 | 含义 |
|---|---|---|
| `num_envs` | 4096 | 并行环境数 |
| `num_observations` | 235 | 观测维度（**必须和 `compute_observations` 实际拼出来的维度一致**） |
| `num_privileged_obs` | `None` | 特权观测维度；None 表示 Critic 也用同样的 obs |
| `num_actions` | 12 | 动作维度 = 关节数 |
| `env_spacing` | 3.0 | 平地时机器人间距（有地形时没用） |
| `send_timeouts` | True | 是否把超时信息传给算法（用于 bootstrapping，见 5.1.8） |
| `episode_length_s` | 20 | 单 episode 最长秒数 |
</details>

<details>
<summary><b>② terrain 地形</b></summary>

| 参数 | 默认 | 含义 |
|---|---|---|
| `mesh_type` | `'trimesh'` | `plane`(平地) / `heightfield`(高度场) / `trimesh`(三角网格) / `none` |
| `horizontal_scale` | 0.1 | 地形网格水平分辨率 [m/像素] |
| `vertical_scale` | 0.005 | 垂直分辨率 [m/单位] |
| `border_size` | 25 | 边界留白 [m]（避免机器人从边缘掉出去） |
| `curriculum` | True | 是否启用由易到难的课程 |
| `measure_heights` | True | 是否把地形高度加入观测（**关掉后 obs 变 48 维**） |
| `measured_points_x/y` | 17×11 个点 | 采样网格（**决定 187 这个数字**） |
| `max_init_terrain_level` | 5 | 初始难度上限 |
| `terrain_length/width` | 8.0 / 8.0 | 每块地形的尺寸 [m] |
| `num_rows` | 10 | 难度等级数 |
| `num_cols` | 20 | 地形类型数 |
| `terrain_proportions` | `[0.1,0.1,0.35,0.25,0.2]` | 各类型占比，见 5.3.2 的坑 |
| `slope_treshold` | 0.75 | 陡于该斜率的面会被"拉直成垂直面"（防止机器人爬陡坡作弊） |
</details>

<details>
<summary><b>③ commands 指令</b></summary>

| 参数 | 默认 | 含义 |
|---|---|---|
| `curriculum` | False | 是否用命令课程（逐渐加大速度范围） |
| `max_curriculum` | 1.0 | 命令课程的速度上限 |
| `num_commands` | 4 | [vx, vy, ωz, heading] |
| `resampling_time` | 10.0 | 每 10 秒换一次指令 |
| `heading_command` | True | **True 时：上层给"目标朝向"，代码自动算出需要的 ωz** |
| `ranges.lin_vel_x` | [-1, 1] | 前后速度范围 [m/s] |
| `ranges.lin_vel_y` | [-1, 1] | 侧移速度范围 [m/s] |
| `ranges.ang_vel_yaw` | [-1, 1] | 转向角速度范围 [rad/s] |
| `ranges.heading` | [-π, π] | 目标朝向范围 |

**`heading_command = True` 的那几行代码**（[327-330](legged_gym/envs/base/legged_robot.py#L327-L330)）：

```python
if self.cfg.commands.heading_command:
    forward = quat_apply(self.base_quat, self.forward_vec)      # 机头指向（世界系）
    heading = torch.atan2(forward[:, 1], forward[:, 0])         # 当前航向角
    self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
```

含义：指令里的第 4 维是"目标航向"，而策略实际吃到的第 3 维是"**当前航向误差 × 0.5，clip 到 ±1**"。这是一个**P 控制器**：朝向误差越大，给的转向速度指令越大。乘以 0.5 是增益。

**为什么这么做？** 因为真机上层导航一般给"目标朝向"而不是"目标角速度"。这样仿真和真机的接口就一致了。（注意这段逻辑在 `_post_physics_step_callback` 里，每步都算，而 `_resample_commands` 只每 10 秒执行一次。）
</details>

<details>
<summary><b>④ init_state 初始状态</b></summary>

| 参数 | A1 值 | 含义 |
|---|---|---|
| `pos` | `[0, 0, 0.42]` | 初始机身位置 [m]，**0.42 是 A1 站立时的机身高度** |
| `rot` | `[0,0,0,1]` | 初始姿态四元数（单位四元数 = 水平） |
| `default_joint_angles` | 12 个关节的角度 | **网络输出全 0 时的姿态**（见 5.1.3 第 2 点） |

> **`pos` 的 z 值高度要小心**：0.42 m 是"站立高度"，但如果给太低（比如 0.2），机器人一开始就插在地里，会被巨大的接触力弹飞；给太高则会自由落体砸地。**迁移新机器人时，这个值应该设为"腿伸直时的一半左右"**，让 PD 控制器能从初始姿态柔和地过渡到站立姿态。
</details>

<details>
<summary><b>⑤ control 控制</b></summary>

| 参数 | A1 值 | 含义 |
|---|---|---|
| `control_type` | `'P'` | 位置 / `'V'` 速度 / `'T'` 力矩 |
| `stiffness` | `{'joint': 20.}` | $K_p$，**用了子串匹配 "joint" 匹配全部 12 个关节** |
| `damping` | `{'joint': 0.5}` | $K_d$ |
| `action_scale` | 0.25 | 动作 → 目标角偏移的比例 |
| `decimation` | 4 | 每个动作跑几个物理步 |
| `use_actuator_network` | False（ANYmal 是 True） | 是否用学习的执行器网络替代 PD |
| `actuator_net_file` | — | 执行器网络权重路径 |

> **⚠️ `stiffness = {'joint': 20.}` 的匹配机制要理解清楚**（[532-541](legged_gym/envs/base/legged_robot.py#L532-L541)）：
>
> ```python
> for dof_name in self.cfg.control.stiffness.keys():
>     if dof_name in name:                     # ★ 子串包含匹配
>         self.p_gains[i] = self.cfg.control.stiffness[dof_name]
>         self.d_gains[i] = self.cfg.control.damping[dof_name]
>         found = True
> ```
>
> - A1 的关节名是 `FL_hip_joint` 等，键 `'joint'` 是它们的子串 → 全部匹配 ✔
> - **如果要用不同的 $K_p$**（人形必须这样，见 8.3），必须用**更具体的、互不重叠的**键，例如 `{'hip': ..., 'knee': ..., 'ankle': ...}`
> - **如果多个键都匹配同一个关节，dict 里后写的会覆盖先写的**（Python 3.7+ dict 保持插入顺序）。所以 `{'joint': 20., 'hip': 80.}` 会让髋关节最终取 80。
> - **如果某个关节一个键都没匹配上，$K_p$ 会被设为 0**，并且打印一条警告。**$K_p = 0$ 意味着那个关节完全没有位置控制能力，机器人会软趴趴**——这是迁移时最容易犯的错，**训练前一定盯着日志看有没有 "PD gain of joint xxx were not defined"**。
</details>

<details>
<summary><b>⑥ asset 模型资源</b></summary>

| 参数 | A1 值 | 含义 |
|---|---|---|
| `file` | `{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf` | URDF 路径（`{LEGGED_GYM_ROOT_DIR}` 会被格式化替换） |
| `name` | `"a1"` | actor 名称 |
| `foot_name` | `"foot"` | **足端 body 的名字子串**，用于算 `feet_indices` |
| `penalize_contacts_on` | `["thigh", "calf"]` | 碰撞惩罚的 body |
| `terminate_after_contacts_on` | `["base"]` | **碰到就"摔倒"的 body** |
| `collapse_fixed_joints` | True | 合并固定关节（A1 的 URDF 里有 10 个固定关节会被合并） |
| `fix_base_link` | False | 是否把机身固定住（机械臂任务用 True） |
| `default_dof_drive_mode` | 3 | 3 = effort（力矩）模式 |
| `self_collisions` | 1 | 1 = 关闭自碰撞，0 = 开启（**位掩码**，不是布尔！） |
| `density` / `armature` / `thickness` | 0.001 / 0 / 0.01 | 默认密度、电机惯量补偿、碰撞体厚度 |
</details>

<details>
<summary><b>⑦ rewards 奖励（见 5.1.6 的完整表）</b></summary>

| 参数 | A1 值 | 含义 |
|---|---|---|
| `only_positive_rewards` | True | 总奖励裁剪到 ≥0 |
| `tracking_sigma` | 0.25 | 速度跟踪奖励的宽容度（见 5.1.6） |
| `soft_dof_pos_limit` | 0.9 | 软限幅比例 |
| `soft_dof_vel_limit` / `soft_torque_limit` | 1.0 | 类似，**但 A1 对应的奖励项是关的** |
| `base_height_target` | 0.25 | 目标机身高度，**但 A1 的 base_height 奖励是关的（⚠️ 设了没用）** |
| `max_contact_force` | 100 | 足端力上限，**A1 的 feet_contact_forces 奖励也是关的** |
</details>

<details>
<summary><b>⑧ normalization 归一化 & noise 噪声</b></summary>

```python
class obs_scales:            # 观测缩放（见 5.1.4）
    lin_vel = 2.0;  ang_vel = 0.25;  dof_pos = 1.0;  dof_vel = 0.05;  height_measurements = 5.0
clip_observations = 100.;  clip_actions = 100.
```

| 噪声参数 | 默认 | 含义 |
|---|---|---|
| `add_noise` | True | 总开关（**play.py 里会关掉**） |
| `noise_level` | 1.0 | 全局缩放 |
| `noise_scales.dof_pos` | 0.01 | 关节角噪声幅值 [rad] |
| `noise_scales.dof_vel` | 1.5 | 关节速度噪声 [rad/s] |
| `noise_scales.lin_vel` | 0.1 | 线速度噪声 [m/s] |
| `noise_scales.ang_vel` | 0.2 | 角速度噪声 [rad/s] |
| `noise_scales.gravity` | 0.05 | 姿态噪声 |
| `noise_scales.height_measurements` | 0.1 | 高度测量噪声 |

**这些值应该怎么定？** 原则是**参照真机传感器的实际精度**。比如真机 IMU 的姿态估计误差大约几度（0.05 rad 合理），编码器的关节角精度约 0.01 rad。**如果你有真机，做 sim2real 时第一件该做的事就是去查这些传感器的 datasheet。**
</details>

<details>
<summary><b>⑨ domain_rand 域随机化</b></summary>

| 参数 | A1 | 含义 |
|---|---|---|
| `randomize_friction` | True | 摩擦随机化开关 |
| `friction_range` | [0.5, 1.25] | 摩擦系数范围 |
| `randomize_base_mass` | **False** | 质量随机化（**A1 是关的，ANYmal-C 是 ±5 kg**） |
| `added_mass_range` | [-1, 1] | 附加质量范围 [kg] |
| `push_robots` | True | 外力推搡开关 |
| `push_interval_s` | 15 | 每 15 秒推一次 → 换算成 750 步 |
| `max_push_vel_xy` | 1.0 | 推搡的瞬时速度 [m/s] |
</details>

<details>
<summary><b>⑩ sim 物理仿真</b></summary>

| 参数 | 默认 | 含义 |
|---|---|---|
| `dt` | 0.005 | **物理步长**（200 Hz）★ |
| `substeps` | 1 | 子步数（**一般不动**） |
| `gravity` | [0, 0, -9.81] | 重力 |
| `up_axis` | 1 | 0 = y 轴朝上，1 = z 轴朝上 |
| `physx.num_threads` | 10 | CPU 线程数 |
| `physx.solver_type` | 1 | 0 = PGS，1 = **TGS**（TGS 更稳，四足必须用） |
| `physx.num_position_iterations` | 4 | 位置求解迭代次数（**接触不稳定就加大到 8**） |
| `physx.contact_offset` | 0.01 | 接触检测距离 [m] |
| `physx.max_gpu_contact_pairs` | 2²³ | **同时存在 8000+ 环境时需加大到 2²⁴** |
| `physx.contact_collection` | 2 | 收集所有子步的接触（**要算碰撞惩罚必须为 2**） |
</details>

<details>
<summary><b>⑪ PPO 训练超参（LeggedRobotCfgPPO）</b></summary>

| 参数 | A1 值 | 含义 |
|---|---|---|
| **policy** | | |
| `init_noise_std` | 1.0 | 初始动作噪声标准差 |
| `actor_hidden_dims` | `[512, 256, 128]` | Actor 三层 MLP |
| `critic_hidden_dims` | `[512, 256, 128]` | Critic 三层 MLP |
| `activation` | `'elu'` | 激活函数 |
| **algorithm** | | |
| `value_loss_coef` | 1.0 | 价值损失权重 |
| `clip_param` | 0.2 | PPO 裁剪范围 |
| `entropy_coef` | 0.01（A1 覆盖） | 熵奖励权重 |
| `num_learning_epochs` | 5 | 每批数据训练几轮 |
| `num_mini_batches` | 4 | 分几个 mini-batch |
| `learning_rate` | 1e-3 | 初始学习率 |
| `schedule` | `'adaptive'` | 按 KL 自动调学习率 ★ |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE 的 λ |
| `desired_kl` | 0.01 | 目标 KL |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| **runner** | | |
| `policy_class_name` | `'ActorCritic'` | 网络类名（支持 `ActorCriticRecurrent`） |
| `algorithm_class_name` | `'PPO'` | 算法类名 |
| `num_steps_per_env` | 24 | 每次迭代采样步数 ★ |
| `max_iterations` | 1500 | 总迭代数 |
| `save_interval` | 50 | 存档间隔 |
| `experiment_name` | `'rough_a1'`（A1 覆盖） | 日志文件夹名 |
| `run_name` | `''` | 子运行名 |
| `resume` / `load_run` / `checkpoint` | False / -1 / -1 | 断点续训 |
</details>

**网络规模量化**（面试可能问到"你的策略网络多大"）：

```
Actor:  235 → 512 → 256 → 128 → 12   ≈ 28.7 万参数（可部署部分）
Critic: 235 → 512 → 256 → 128 → 1    ≈ 28.5 万参数（只训练时用）
std:    12 个可学习标量
```

**完整训练用的总参数量 ≈ 57 万**，但**导出到真机的 JIT 策略只有 Actor 的 28.7 万**（[helpers.py:180-190](legged_gym/utils/helpers.py#L180-L190) 只 deepcopy 了 `actor_critic.actor`）。**这个"训练大、部署小"的思路值得记住：Critic 可以很大很贵，因为它不用上真机。**

### 5.3 terrain.py 地形生成

#### 5.3.1 全过程：从配置到物理网格

```python
# terrain.py:38-73
class Terrain:
    def __init__(self, cfg, num_robots):
        self.env_length = cfg.terrain_length      # 8 m
        self.env_width  = cfg.terrain_width       # 8 m
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(...))]  # ★ 累积和
        self.width_per_env_pixels  = int(env_width  / horizontal_scale)   # 8/0.1 = 80 像素
        self.length_per_env_pixels = int(env_length / horizontal_scale)   # 80
        self.border = int(border_size / horizontal_scale)                 # 25/0.1 = 250 像素
        self.tot_cols = num_cols * width_per_env_pixels  + 2*border       # 20*80 + 500 = 2100
        self.tot_rows = num_rows * length_per_env_pixels + 2*border       # 10*80 + 500 = 1300
        self.height_field_raw = np.zeros((tot_rows, tot_cols), dtype=np.int16)   # ★ 整张高度图
        if cfg.curriculum:  self.curiculum()                              # ★ 生成
        self.heightsamples = self.height_field_raw
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(...)  # ★ 转三角网格
```

**理解这张图**：整张地形是一张 1300×2100 的**整数高度图**（int16，1 单位 = `vertical_scale` = 5 mm），在它上面划分成 10 行 × 20 列共 200 块 8m×8m 的地形块。外圈留 250 像素（25 m）的边界，防止机器人跑出地图掉下去。

**为什么用 int16 高度图而不是直接建网格？** 因为**程序化生成快且省内存**：1300×2100 的 int16 数组只有 5.5 MB，而如果直接建三角网格，顶点数会是千万级。最后只在需要时用 `convert_heightfield_to_trimesh` 一次性转换。

#### 5.3.2 `curiculum()` + `make_terrain()`：难度和类型怎么编码

```python
# terrain.py:85-92
def curiculum(self):
    for j in range(self.cfg.num_cols):          # 20 种地形类型
        for i in range(self.cfg.num_rows):      # 10 个难度等级
            difficulty = i / self.cfg.num_rows        # 0.0 → 0.9
            choice     = j / self.cfg.num_cols + 0.001 # 0.05 → 0.95
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
```

**这是一个非常聪明的"二维编码"**：
- **行号 i → 难度 `difficulty`**（0 到 0.9 线性递增）
- **列号 j → 地形类型 `choice`**（0 到 0.95 映射到不同类型）

所以 **`(i, j)` 一个坐标就同时编码了"多难"和"什么类型"**。地形课程只需要改 `terrain_levels`（行号）就能让机器人"在同一类型上逐渐变难"。

`make_terrain`（[109-145](legged_gym/utils/terrain.py#L109-L145)）用 `choice` 落在 `proportions`（累积和）的哪个区间决定类型：

先算 `proportions`（默认 5 种类型 `[0.1, 0.1, 0.35, 0.25, 0.2]`）：

```
proportions = [0.1, 0.2, 0.55, 0.8, 1.0]      # 累积和
```

| choice 区间 | 地形类型 | 关键生成参数（随 difficulty 变化） | 占比 |
|---|---|---|---|
| `[0, 0.1)` | **光滑斜坡** | `slope = ±difficulty × 0.4`（一半是下坡） | 10% |
| `[0.1, 0.2)` | **粗糙斜坡** | 斜坡 + 均匀随机噪声 ±5 cm | 10% |
| `[0.2, 0.55)` | **上楼梯** | `step_height = 0.05 + 0.18×difficulty` | 35% |
| `[0.55, 0.8)` | **下楼梯** | 同上取负 | 25% |
| `[0.8, 1.0)` | **离散障碍物** | 高 `0.05 + 0.2×difficulty`，20 个随机矩形 | 20% |

**难度是怎么随 difficulty 增长的**（[115-121](legged_gym/utils/terrain.py#L115-L121)）：

```python
slope = difficulty * 0.4                                        # 最高 0.4 的斜率 ≈ 21.8°
step_height = 0.05 + 0.18 * difficulty                          # 台阶从 5 cm 到 23 cm
discrete_obstacles_height = 0.05 + difficulty * 0.2             # 障碍高 5→25 cm
stepping_stones_size = 1.5 * (1.05 - difficulty)                # 落石从大到小
gap_size = 1. * difficulty ;  pit_depth = 1. * difficulty       # 坑/沟深 0→1 m
```

以 A1 的体长（约 0.5 m）和站立高度（0.42 m）来看，**最高难度 23 cm 的台阶和 1 m 的坑对 A1 已经相当难了**——这也是为什么训练需要 1.5 亿步。

> **⚠️ 重要坑：`make_terrain` 里有 8 个分支，但默认配置只够 5 个。**
>
> 代码里后面还有 `stepping_stones_terrain`（落石）、`gap_terrain`（沟）、`pit_terrain`（坑）三个分支，它们访问 `proportions[5]`、`proportions[6]`，而**默认 5 个比例只产生下标 0~4**。之所以不报 `IndexError`，是因为 `choice` 最大只有 0.951 < `proportions[4]` = 1.0，**永远走不到那几个分支**——它们实际上是**死代码**。
>
> **想让 A1 走落石/过沟怎么办？** 把 `terrain_proportions` 扩充成 8 个，比如：
> ```python
> terrain_proportions = [0.05, 0.05, 0.15, 0.15, 0.15, 0.2, 0.1, 0.15]
> #                                                    ↑落石 ↑沟 ↑坑
> ```
> 这个"读了源码才发现分支走不到"的体验，也是很好的面试素材。

#### 5.3.3 `add_terrain_to_map`：拼图和出生点计算（[147-164](legged_gym/utils/terrain.py#L147-L164)）

```python
start_x = self.border + i * self.length_per_env_pixels     # 计算这块地形在大图里的位置
end_x   = self.border + (i + 1) * self.length_per_env_pixels
self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw   # ★ 贴上去

# 出生点：这块地形的中心
env_origin_x = (i + 0.5) * self.env_length
env_origin_y = (j + 0.5) * self.env_width
# 出生点高度：中心 2×2 m 范围内的最高点（保证不会生成在坑里）
x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * terrain.vertical_scale
self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
```

**注意 `env_origins` 的 z 取的是"中心 2m×2m 区域的最高点"**——这样保证机器人不会出生在坑里或台阶中间，一开始就能稳定站住。这是一个容易被忽略但很关键的细节。

### 5.4 task_registry.py 与 helpers.py

#### 5.4.1 注册表模式（[task_registry.py:44-63](legged_gym/utils/task_registry.py#L44-L63)）

```python
class TaskRegistry():
    def __init__(self):
        self.task_classes, self.env_cfgs, self.train_cfgs = {}, {}, {}

    def register(self, name, task_class, env_cfg, train_cfg):     # ★ 三个东西绑在一起
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

task_registry = TaskRegistry()      # ★ 全局单例
```

三个字典的键都是任务名字符串。**注册发生在 [envs/__init__.py:45-51](legged_gym/envs/__init__.py#L45-L51)**：

```python
task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("anymal_c_flat",  Anymal, AnymalCFlatCfg(),  AnymalCFlatCfgPPO())
task_registry.register("anymal_b",       Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO())
task_registry.register("a1",             LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())    # ★ 注意是 LeggedRobot
task_registry.register("cassie",         Cassie, CassieRoughCfg(),  CassieRoughCfgPPO())
```

**关键观察：A1 注册的是 `LeggedRobot`（基类），不是子类。** 因为 A1 没有额外功能（不需要执行器网络），直接复用基类就够了。**ANYmal-C 注册的是 `Anymal`**，因为它在 [anymal.py](legged_gym/envs/anymal_c/anymal.py) 里覆写了 `_compute_torques` 来加载执行器网络。

**这就是"任务注册表"模式的全部价值**：新增一个机器人 = 写一个配置 + 加一行 `register`。**这就是你第 8 章要干的事。**

#### 5.4.2 `Anymal` 的执行器网络（[anymal.py:63-80](legged_gym/envs/anymal_c/anymal.py#L63-L80)）—— 值得单独看

```python
def _compute_torques(self, actions):
    if self.cfg.control.use_actuator_network:
        with torch.inference_mode():
            self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
            self.sea_input[:, 0, 1] = self.dof_vel.flatten()
            torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(
                self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
        return torques
    else:
        return super()._compute_torques(actions)     # ★ 退回 PD
```

**它在做什么？** 理想 PD 控制器 $K_p e - K_d \dot q$ 是高线性的"假电机"。真机的电机有**延迟、饱和、摩擦、非线性**。ANYmal 的解法是：

1. 在真机上采集大量 `(位置误差, 速度) → 实测力矩` 的数据
2. 训一个 LSTM 网络拟合这个映射
3. 在仿真里用这个 LSTM 替代 PD

**为什么用 LSTM 而不是 MLP？** 因为电机有**迟滞和热效应**——同样的输入，力矩取决于历史。LSTM 的隐藏状态能编码这个历史（`sea_hidden_state` / `sea_cell_state` 在每条轨迹开始时清零，见 [anymal.py:47-51](legged_gym/envs/anymal_c/anymal.py#L47-L51)）。

**这是 sim2real 三大支柱的第三根**（前两根是观测噪声和域随机化）。**但注意 A1 没有这个**——A1 用的是纯 PD。如果你要做 A1 的真机迁移，可以考虑：
- 给 A1 也训一个执行器网络（需要真机数据）
- 或者更实际的做法：在 PD 之后加一层"电机延迟 + 力矩噪声"，用更简单的方式近似真机特性

#### 5.4.3 helpers.py 的四个关键函数

| 函数 | 作用 | 为什么重要 |
|---|---|---|
| `class_to_dict` [:41](legged_gym/utils/helpers.py#L41) | 嵌套配置类 → 字典 | 让 `reward_scales.pop()`、`**kwargs` 展开能工作 |
| `get_args` [:152](legged_gym/utils/helpers.py#L152) | 命令行参数 → `args` | 见 7.2 的参数表 |
| `update_cfg_from_args` [:127](legged_gym/utils/helpers.py#L127) | 命令行覆盖配置 | **优先级：命令行 > 配置文件** |
| `get_load_path` [:103](legged_gym/utils/helpers.py#L103) | 找最后一个 checkpoint | `--load_run=-1` 和 `--checkpoint=-1` 的语义在这 |
| `export_policy_as_jit` [:180](legged_gym/utils/helpers.py#L180) | 导出可部署策略 | 见下 |

**`export_policy_as_jit` 是"从仿真到真机"的最后一步**：

```python
def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):            # LSTM 版本
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)                         # → policy_lstm_1.pt
    else:                                            # MLP 版本
        model = copy.deepcopy(actor_critic.actor).to('cpu')   # ★ 只要 actor，不要 critic
        traced_script_module = torch.jit.script(model)        # ★ TorchScript 编译
        traced_script_module.save(path)                        # → policy_1.pt
```

**三个关键点**：
1. **只导出 actor**（`actor_critic.actor`）——critic 在部署时毫无用处
2. **`.to('cpu')`** ——真机（如 Jetson、机载工控机）通常没有 CUDA，或者不希望你占用 GPU
3. **`torch.jit.script`** ——编译成 TorchScript，可以在 **C++ 里直接加载运行**（不需要 Python 环境）。真机上的部署程序一般用 C++ 写，所以这一步是必需的。

**你的 `logs/rough_a1/exported/policies/policy_1.pt` 就是这样一个可以直接在 C++ 里 `torch::jit::load()` 加载的策略**——只有 28.7 万参数，约 1.1 MB。**这是一个可以在面试里讲的具体数字。**

> 注意 `play.py:78` 用 `policy(obs.detach())` 调用，而 `get_inference_policy` 返回的是 `actor_critic.act_inference`（[on_policy_runner.py:233](rsl_rl/rsl_rl/runners/on_policy_runner.py#L233)）——**推理时用的是动作均值，不是采样值**（见 2.2 节）。

### 5.5 rsl_rl：PPO 的实现

#### 5.5.1 `ActorCritic`：网络结构（[actor_critic.py:38-136](rsl_rl/rsl_rl/modules/actor_critic.py#L38-L136)）

```python
# 38-88  构建两个独立的 MLP
self.actor  = nn.Sequential(Linear(235, 512), ELU, Linear(512, 256), ELU, Linear(256, 128), ELU, Linear(128, 12))
self.critic = nn.Sequential(Linear(235, 512), ELU, Linear(512, 256), ELU, Linear(256, 128), ELU, Linear(128, 1))
self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))    # ★ 12 个可学习的噪声标准差
Normal.set_default_validate_args = False                              # ★ 关掉 torch 的参数检查，提速
```

**为什么 Actor 和 Critic 完全分开、不共享底层？** 因为它们的输入可以不同（非对称训练，见 2.12），而且**共享底层会让两个目标的梯度互相干扰**。这个仓库选了完全分离——参数多一倍，但训练更稳。

**这个网络小得惊人**：28 万参数的 MLP，就学会了一只机器狗的行走。**这本身就是个值得讲的点**——"不需要大模型，一个 3 层 MLP 就够了，因为任务结构已经被观测和控制器的设计很好地编码了"。对比一下：GPT-2 有 15 亿参数。

```python
# 123-132
def act(self, observations, **kwargs):
    self.update_distribution(observations)
    return self.distribution.sample()          # ★ 训练：采样

def act_inference(self, observations):
    return self.actor(observations)            # ★ 部署：直接用均值
```

#### 5.5.2 `PPO.update()`：全部的损失计算（[ppo.py:120-187](rsl_rl/rsl_rl/algorithms/ppo.py#L120-L187)）

```python
for obs_batch, ... in generator:              # 20 个 mini-batch（5 epochs × 4 batches）
    self.actor_critic.act(obs_batch, ...)     # 用新策略重新过一遍这批数据
    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
    value_batch = self.actor_critic.evaluate(critic_obs_batch, ...)

    # ---- KL 自适应学习率（见 2.7）----
    if self.schedule == 'adaptive':
        kl = torch.sum(torch.log(sigma/sigma_old + 1e-5) + (sigma_old² + (mu_old-mu)²)/(2·sigma²) - 0.5, axis=-1)
        ...调节 self.learning_rate...

    # ---- ① 策略损失：PPO 裁剪（见 2.6）----
    ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
    surrogate_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 0.8, 1.2)).mean()

    # ---- ② 价值损失：也裁剪 ----
    value_clipped = target_values + (value_batch - target_values).clamp(-0.2, 0.2)
    value_loss = torch.max((value_batch - returns)², (value_clipped - returns)²).mean()

    # ---- ③ 总损失 = 策略 + 价值 - 熵 ----
    loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

    self.optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)   # 梯度裁剪
    self.optimizer.step()
```

**三件事串起来就是 PPO 的全部**：

$$L = L^{clip}_{policy} + c_1 \cdot L^{clipped}_{value} - c_2 \cdot H[\pi]$$

**`get_actions_log_prob` 的用法要注意**（[actor_critic.py:127-128](rsl_rl/rsl_rl/modules/actor_critic.py#L127-L128)）：

```python
def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)      # ★ 12 个维度求和
```

因为 12 个关节的动作是**独立**高斯（对角协方差），所以联合对数概率 = 各维之和。**如果你要改成考虑关节相关性（协方差矩阵），这个模型就不够用了**——但实践证明对角高斯足够，而且计算便宜得多。

#### 5.5.3 `RolloutStorage`：经验池（[rollout_storage.py:53-137](rsl_rl/rsl_rl/storage/rollout_storage.py#L53-L137)）

**张量形状（A1）**：

```python
self.observations = torch.zeros(24, 4096, 235)     # [num_steps, num_envs, obs_dim]
self.actions      = torch.zeros(24, 4096, 12)
self.rewards      = torch.zeros(24, 4096, 1)
self.dones        = torch.zeros(24, 4096, 1).byte()
self.values / returns / advantages = torch.zeros(24, 4096, 1)
self.mu / sigma   = torch.zeros(24, 4096, 12)
```

**布局是"时间优先"`[T, N, D]` 而不是 `[N, T, D]`。** 为什么？因为 GAE 需要**从最后一步往前迭代**，时间在前能让 `self.values[step+1]` 这样的切片是连续的。

**总内存占用估算**：
```
观测: 24×4096×235×4 B = 92 MB
动作/μ/σ: 3 × 24×4096×12×4 B = 14 MB
values/returns/advantages/logprob: 4 × 24×4096×4 B = 1.6 MB
合计约 110 MB
```
**这只是 rollout buffer，还没算 4096 个环境的物理仿真状态和网络本身。** 所以 4096 环境大概需要 16~24 GB 显存（视地形复杂度和 Isaac Gym 版本）。**显存不够时第一个该调的就是 `num_envs`。**

`mini_batch_generator`（[147-183](rsl_rl/rsl_rl/storage/rollout_storage.py#L147-L183)）：

```python
batch_size = num_envs * num_transitions_per_env          # 4096 × 24 = 98304
mini_batch_size = batch_size // num_mini_batches         # 98304 / 4 = 24576
indices = torch.randperm(...)                             # ★ 随机打乱

observations = self.observations.flatten(0, 1)            # ★ [24, 4096, 235] → [98304, 235]
for epoch in range(num_epochs):        # 5 轮
    for i in range(num_mini_batches):  # 每轮 4 个 batch
        yield ...                       # 共 20 次梯度更新
```

**`flatten(0, 1)` 把时间和环境两个维度压平**——因为 PPO 更新**不关心时序**（GAE 已经在 `compute_returns` 里算完了优势值，更新时每个样本都是独立的）。**这正是 PPO 简单高效的原因：不需要 RNN 那样处理时序依赖。**

> **注意一个细节**：`flatten(0,1)` 会得到 `[T, N, D]` 展平成 `[T*N, D]`，也就是说**同一个时间步的所有环境连在一起**。理论上随机打乱后无所谓，但理解这个顺序有助于你 debug。

---

## 6. 训练全链路时序图

把前面所有内容连起来，**从你敲下回车到机器狗走路**：

```
$ python legged_gym/scripts/train.py --task=a1 --headless --num_envs=2048 --max_iterations=1500
   │
   ├─[1] get_args()                          解析命令行 → args.task="a1", args.num_envs=2048
   │
   ├─[2] make_env(args.task)  ──── 造世界 ────────────────────────────────────────┐
   │      ├─ get_cfgs("a1")            取出 A1RoughCfg + A1RoughCfgPPO            │
   │      ├─ update_cfg_from_args()    num_envs: 4096 → 2048（命令行覆盖）        │
   │      ├─ set_seed(1)               固定种子                                  │
   │      ├─ parse_sim_params()        cfg.sim → gymapi.SimParams                │
   │      └─ LeggedRobot(...)                                                     │
   │           ├─ _parse_cfg()         dt=0.02, max_episode_length=1000           │
   │           ├─ BaseTask.__init__    acquire_gym, 分配 obs/rew/reset 缓冲        │
   │           ├─ create_sim()                                                    │
   │           │    ├─ create_sim()    GPU 上创建物理世界                          │
   │           │    ├─ Terrain()       ★ 生成 1300×2100 高度图 → 200 块地形       │
   │           │    ├─ _create_trimesh()  转成三角网格提交给 PhysX                │
   │           │    └─ _create_envs()  ★ 循环 2048 次：                            │
   │           │         load_asset(a1.urdf)     解析 URDF                       │
   │           │         _process_rigid_shape_props()  摩擦随机化（64 档分桶）     │
   │           │         _process_dof_props()          读关节限幅（髋20/大腿小腿55）│
   │           │         _process_rigid_body_props()   质量随机化（A1 关闭）       │
   │           │         名字匹配 → feet_indices / penalised / termination 索引    │
   │           ├─ _init_buffers()      ★ 把仿真张量接成 PyTorch 张量              │
   │           │                        读 default_joint_angles → default_dof_pos │
   │           │                        子串匹配 stiffness → p_gains / d_gains    │
   │           └─ _prepare_reward_function()  ★ 10 个生效奖励 → 函数指针列表       │
   │                                                                              │
   ├─[3] make_alg_runner(env)  ──── 造大脑 ───────────────────────────────────────┤
   │      ├─ log_dir = logs/rough_a1/Sep21_13-29-25_/                             │
   │      ├─ ActorCritic(235, 235, 12, [512,256,128], elu)   ★ 28.7万+28.5万参数  │
   │      ├─ PPO(actor_critic, lr=1e-3, clip=0.2, γ=0.99, λ=0.95, kl=0.01)        │
   │      ├─ init_storage(2048, 24, [235], [None], [12])  ★ ~55 MB                │
   │      └─ env.reset()              2048 只狗在各难度地形上就位                   │
   │                                                                              │
   └─[4] runner.learn(1500)  ──── 开训 ──────────────────────────────────────────┘
          │
          └─ for it in 0..1499:  ┌───────────────────────────────────────────────┐
                │                │ 【采样阶段】 num_steps_per_env = 24 步         │
                │                │  for i in 0..23:                              │
                │                │    act()      策略前向 → N(μ,σ) 采样 → 12 维动作 │
                │                │    env.step() ┌──────────────────────────────┐ │
                │                │              │ clip 到 ±100                  │ │
                │                │              │ for _ in range(4):  ← decimation│
                │                │              │   _compute_torques()          │ │
                │                │              │   τ = 20·(0.25a + q_d − q) − 0.5·q̇│
                │                │              │   clip 到 ±20/±55             │ │
                │                │              │   simulate(0.005s)            │ │
                │                │              │ post_physics_step():          │ │
                │                │              │   ⑦ → 机体系速度/投影重力       │ │
                │                │              │   ③ → 重采样命令/测地形/推狗    │ │
                │                │              │   ④ → 机身触地? → reset       │ │
                │                │              │   ⑤ → 10 项奖励加权求和        │ │
                │                │              │   ⑥ → 重置摔了的（随机初始态）  │ │
                │                │              │   ⑦ → 拼 235 维观测 + 加噪声    │ │
                │                │              └──────────────────────────────┘ │
                │                │    process_env_step()  → 存进 storage         │
                │                │  compute_returns()     → GAE 反向递推          │
                │                ├───────────────────────────────────────────────┤
                │                │ 【学习阶段】 5 epochs × 4 mini-batches          │
                │                │  for 20 次:                                    │
                │                │    ratio = π_new/π_old → clamp [0.8, 1.2]       │
                │                │    loss = 策略损失 + 价值损失 − 熵              │
                │                │    KL > 0.02? → lr /= 1.5  (自适应)             │
                │                │    反传 → 梯度裁剪(1.0) → Adam 更新             │
                │                └───────────────────────────────────────────────┘
                │
                ├─ 每 50 次迭代：save(model_{it}.pt)
                └─ tensorboard 记录所有曲线
                         ↓
              1500 次迭代后 → model_1500.pt
                         ↓
$ python legged_gym/scripts/play.py --task=a1
   ├─ 覆盖配置：num_envs≤50, 地形 5×5, 关噪声/关摩擦随机/关推搡
   ├─ resume=True → 自动找 logs/rough_a1/ 下最后一个 run 的最后一个 model
   ├─ policy = runner.get_inference_policy()     ★ 用 act_inference（均值）
   ├─ export_policy_as_jit() → logs/rough_a1/exported/policies/policy_1.pt   ★ 可部署
   └─ 跑 10 个 episode，记录状态曲线（Logger 画图）
```

---

## 7. 从零跑通 + 踩坑排错

### 7.1 环境搭建（关键：版本必须对齐）

这个仓库对版本极其敏感，**不是随便装个新版本就能跑**。

```
Python 3.8（3.6/3.7 也行，强烈建议 3.8）
CUDA 11.3+（Isaac Gym 需要）
PyTorch 1.10.0 + cu113
Isaac Gym Preview 3 或 Preview 4   ← ⚠️ Preview 2 一定不行
rsl_rl v1.0.2                       ← ⚠️ 新版 rsl_rl 接口已变，会报错
```

**安装顺序**：

```bash
# 1. 建虚拟环境（强烈建议用 conda/mamba，别用系统 Python）
conda create -n legged python=3.8 -y && conda activate legged

# 2. PyTorch（版本必须匹配）
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 必须 True

# 3. Isaac Gym
#    从 https://developer.nvidia.com/isaac-gym 下载 Preview 3/4
cd isaacgym/python && pip install -e .
cd examples && python 1080_balls_of_solitude.py     # ★ 官方冒烟测试，必须能跑

# 4. rsl_rl（❗必须 checkout 到 v1.0.2）
#    本仓库里已经有 rsl_rl/ 目录了，直接用：
cd /home/quinn/文档/legged_gym/rsl_rl && pip install -e .
#    如果你想用官方仓库：
#    git clone https://github.com/leggedrobotics/rsl_rl && cd rsl_rl && git checkout v1.0.2 && pip install -e .

# 5. legged_gym
cd /home/quinn/文档/legged_gym && pip install -e .
```

**⚠️ 关于 Isaac Gym 的现实问题（你必须知道）**：
NVIDIA 已经在 2024 年把重心转向 **Isaac Sim / Isaac Lab**，README 里也明确写了"本仓库将只接受有限更新，建议迁移到 Isaac Lab"。Isaac Gym 的官方下载页可能已经关闭或需要老账号。**几条现实路径**：

| 路径 | 优点 | 缺点 |
|---|---|---|
| 继续用 Isaac Gym（能下到的话） | 与 legged_gym 完全匹配，教程最多 | 官方已停止维护，新显卡可能不兼容 |
| 迁移到 **Isaac Lab**（官方后继者） | 官方支持，功能更强 | API 大改，学习曲线陡 |
| 用 **legged_gym 的社区分支** | 保留 Isaac Gym 体验且有新机器人 | 质量参差 |
| 用 MuJoCo + 其他 RL 库 | 开源友好，不依赖 CUDA | 并行规模小，训练慢很多 |

**建议**：先把这套 legged_gym 跑通（理解原理），**同时关注 Isaac Lab**——因为 2026 年找工作，"熟悉 Isaac Lab"比"熟悉 Isaac Gym"更有市场。**但两者概念是通的**（观测/奖励/域随机化/课程学习的设计思路完全一致），所以你在 legged_gym 上学到的东西不会浪费。

### 7.2 命令速查

```bash
# ---------- 训练 ----------
# 基础训练（A1 崎岖地形）
python legged_gym/scripts/train.py --task=a1
# 无渲染（★ 训练时必加，否则慢好几倍）
python legged_gym/scripts/train.py --task=a1 --headless
# 指定环境数和迭代数（显存不够时用）
python legged_gym/scripts/train.py --task=a1 --headless --num_envs=2048 --max_iterations=1000
# 用 CPU 训练（调试用，极慢）
python legged_gym/scripts/train.py --task=a1 --sim_device=cpu --rl_device=cpu
# 自定义实验名（便于区分实验）
python legged_gym/scripts/train.py --task=a1 --headless --experiment_name=debug --run_name=lr5e4
# 断点续训
python legged_gym/scripts/train.py --task=a1 --resume --load_run=-1 --checkpoint=-1
# 指定随机种子
python legged_gym/scripts/train.py --task=a1 --headless --seed=42

# ---------- 回放 / 评估 ----------
python legged_gym/scripts/play.py --task=a1
# 加载特定 checkpoint（改 play.py 里的 train_cfg.runner.load_run / checkpoint，或用 --checkpoint）
python legged_gym/scripts/play.py --task=a1 --checkpoint=500

# ---------- 查看训练曲线 ----------
tensorboard --logdir=logs/rough_a1
# 浏览器打开 http://localhost:6006

# ---------- 冒烟测试（改完代码先跑这个） ----------
python legged_gym/tests/test_env.py --task=a1 --headless
```

**训练时的键盘操作**：按 `v` 可以**切换渲染开关**。**训练时务必按 `v` 关掉渲染**——渲染会占据大量 GPU 资源，让你损失数倍的训练速度（README 里专门强调了这点）。

### 7.3 性能与显存调优

| 症状 | 原因 | 对策 |
|---|---|---|
| 显存不足 OOM | 环境数太多 / rollout buffer 太大 | ① `--num_envs=1024`；② 减小 `num_steps_per_env`；③ 简化地形（`num_rows/num_cols` 改小） |
| 训练慢 | 渲染开着 | **按 `v` 关渲染，或用 `--headless`** |
| 训练慢 | 环境数太少 | GPU 没吃饱，加大 `num_envs` |
| 训练慢 | CPU 瓶颈 | 加大 `sim.physx.num_threads` |
| 超过 8000 环境时报接触错误 | PhysX 接触对上限 | `max_gpu_contact_pairs = 2**24` |
| 接触不稳定、机器人抖动穿透 | 求解器迭代不够 | `num_position_iterations` 4 → 8；确认 `solver_type = 1`（TGS） |
| 训练速度不到 ~5000 steps/s | 配置不合理 | 正常 A1 4096 环境在 3090 上应达到 2~5 万 steps/s |

**实测参考（写简历时能用）**：A1，4096 环境，RTX 3090，崎岖地形，约 **2~5 万 steps/s**，1500 次迭代（1.5 亿步）大约 **1~2 小时**。ANYmal-C 平地版只需 300 次迭代，约 **10~20 分钟**——这就是论文标题"Learning to Walk in **Minutes**"的由来。

### 7.4 训练不收敛排查清单

**按这个顺序排查**（从最常见到最少见）：

#### 第一层：看 tensorboard 曲线判断症状

| 曲线表现 | 诊断 | 对策 |
|---|---|---|
| `Train/mean_reward` 完全不动 | 奖励信号有问题 | 见下面第二层 |
| 奖励涨一下就崩，反复震荡 | 学习率太大 / KL 失控 | 看 `Loss/learning_rate`，初始 lr 降到 5e-4；`desired_kl` 从 0.01 调到 0.005 |
| 奖励缓慢上升但很慢 | 探索不足 / 学习率太小 | `init_noise_std` 加大；`entropy_coef` 加大 |
| `Policy/mean_noise_std` 不下降 | 策略没收敛到确定性解 | 加大 `entropy_coef`？不对——**降小** `entropy_coef`；检查奖励是否矛盾 |
| `Train/mean_episode_length` 很低 | 老是摔倒 | 见"摔倒早"一行 |
| `mean_episode_length` 很快就满 1000 | 机器人学会了"站着不动" | 检查 `stand_still` 和 `only_positive_rewards`；看 `rew_tracking_lin_vel` 是否为 0 |
| `value_function loss` 爆炸 | 奖励量级太大 | 检查奖励尺度；`value_loss_coef` 调小；确认优势标准化在起作用 |

#### 第二层：奖励与终止

| 症状 | 可能原因 | 对策 |
|---|---|---|
| 完全学不会 | **`num_observations` 和实际 obs 维度不匹配** | ★ 最常见的错误！打印 `env.obs_buf.shape`，对照 5.1.4 的表重算 |
| 完全学不会 | **`default_joint_angles` 姿态不合理** | 在 `play.py` 里看初始姿态是否像个能站的样子 |
| 完全学不会 | **`stiffness` 的键没匹配上任何关节** | ★ 检查日志里有没有 "PD gain of joint xxx were not defined" |
| 机器人"劈叉"或"趴地" | $K_p$ 太小 / 目标姿态不合理 | 加大 `stiffness`；检查 `default_joint_angles` |
| 一出场就摔倒 | `terminate_after_contacts_on` 匹配到了不该匹配的 body | ★ 子串匹配的坑，见 5.1.8；打印 `env.termination_contact_indices` 对应的名字 |
| 一出场就摔倒 | 初始高度 `init_state.pos` 不对 | 调到合理站立高度 |
| 机器人抽搐抖动 | `action_rate` / `dof_acc` 惩罚太小 | 加大这两项；检查 `d_gains` 是否太小 |
| 站不住、一直弹跳 | `lin_vel_z` 惩罚太小 | 加大 `lin_vel_z`（A1 已经是 -2.0） |
| 不会走路只原地踏步 | `tracking_lin_vel` 权重相对其他项太小 | 检查分项奖励曲线 `rew_*`，看哪一项在主导 |
| 步态很碎很小 | `feet_air_time` 太小 | 加大；注意 `(air_time - 0.5)` 的阈值 |

> **★ 最重要的调试技巧：看 tensorboard 里的分项奖励 `Episode/rew_*`。**
> 总奖励涨了不等于学对了。如果 `rew_lin_vel_z`（惩罚）在涨，说明机器人在疯狂弹跳却在"骗分"。
> **每项奖励都应该单独看**：正奖励项应该上升，负奖励项应该趋于 0。

#### 第三层：PPO 超参

| 症状 | 调什么 | 方向 |
|---|---|---|
| 训练震荡不稳定 | `learning_rate` | 1e-3 → 5e-4 |
| 更新幅度太大 | `clip_param` | 0.2 → 0.1 |
| 更新幅度太大 | `desired_kl` | 0.01 → 0.005 |
| 学得太慢 | `num_steps_per_env` | 24 → 32（更多数据） |
| 一批数据学不透 | `num_learning_epochs` | 5 → 8（但过大会过拟合旧数据） |
| 过早收敛到局部最优 | `entropy_coef` | 0.01 → 0.02 |
| 策略太保守 | `init_noise_std` | 1.0 → 1.5 |

#### 第四层：环境设置

| 症状 | 原因 | 对策 |
|---|---|---|
| 训练初期就全军覆没 | 地形太难 | 关课程：`terrain.curriculum = False` + `selected_terrain` 只用平地 |
| 学得快但一换地形就废 | 地形太单一 | 加大 `terrain_proportions` 的多样性 |
| 仿真里好好的，真机不行 | 缺少 sim2real 机制 | 打开 `randomize_friction` / `randomize_base_mass` / `push_robots`；加大噪声 |

### 7.5 常见报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `Task with name: xxx was not registered` | 忘了 `register` 或忘了 `from legged_gym.envs import *` | 检查 [envs/__init__.py](legged_gym/envs/__init__.py) |
| `'LeggedRobot' object has no attribute '_reward_xxx'` | ★ 奖励名和方法名对不上 | grep 核对，见 5.1.7 |
| `AssertionError: Rollout buffer overflow` | `num_transitions_per_env` 与实际步数不符 | 一般是你改了 `num_steps_per_env` 却在中途改回去了 |
| `CUDA out of memory` | 见 7.3 | — |
| `AttributeError: module 'numpy' has no attribute 'float'` | numpy 版本太新（≥1.24） | `pip install "numpy<1.24"` |
| Isaac Gym 导入时报 GL / Vulkan 错误 | 无显示设备 | 加 `--headless` |
| 找不到 `.dae` 网格文件 | URDF 里的 mesh 路径不对 | 见 8.2 节 |
| `RuntimeError: shape '[4096, 0]' is invalid` | `num_dof = 0`，资产加载失败 | 检查 URDF 能否被解析 |

---

## 8. 迁移到自己的机器人（G1 / 天工 3.0 / Go2）

> 这是你寒假实习前最值得做的一件事——**因为"我在 legged_gym 上加过新机器人"比"我跑通过官方 demo"高一个档次**。而且你已经明确想搞 G1、天工 3.0、Go2。

### 8.1 四步流程（README 的官方说法 + 我的补充）

**第 0 步：准备资产（最花时间，占 60% 工作量）**
```
resources/robots/<your_robot>/
├── urdf/<your_robot>.urdf
└── meshes/*.STL (或 .dae/.obj)
```

**第 1 步：写配置** `legged_gym/envs/<your_robot>/<your_robot>_config.py`
继承现有的配置类，只覆盖差异。**参考 [a1_config.py](legged_gym/envs/a1/a1_config.py) 的结构**，必须覆盖：

```python
class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = ???        # ★★ 必须重算！见 8.4
        num_actions = ???             # ★ = 关节数
    class init_state(...):
        pos = [0., 0., ???]           # ★ 站立高度
        default_joint_angles = {...}  # ★★★ 膝/髋的弯曲姿态
    class control(...):
        stiffness = {...}             # ★★★ 按关节类型分组
        damping = {...}
        action_scale = ???
    class asset(...):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/xxx/urdf/xxx.urdf"
        name = "xxx"
        foot_name = "foot"            # ★ 按你的 URDF 改
        terminate_after_contacts_on = ["trunk"]   # ★ 人形是 trunk/pelvis
    class rewards(...):
        class scales(MyRobotCfg.rewards.scales):
            ...                       # ★ 打开你要的奖励
```

**第 2 步：（可选）写环境类** `legged_gym/envs/<your_robot>/<your_robot>.py`
只有当你要覆写方法时才需要（比如加执行器网络、加自定义奖励）。**如果不需要，直接复用 `LeggedRobot`，像 A1 那样。**

**第 3 步：注册** 在 [envs/__init__.py](legged_gym/envs/__init__.py) 加一行
```python
from .my_robot.my_robot_config import MyRobotCfg, MyRobotCfgPPO
task_registry.register("my_robot", LeggedRobot, MyRobotCfg(), MyRobotCfgPPO())
```

**第 4 步：冒烟测试 → 小规模训练 → 调参**
```bash
# ① 先测试环境能不能创建（改 envs 里的 num_envs 为 32）
python legged_gym/tests/test_env.py --task=my_robot --headless
# ② 可视化看初始姿态对不对（不开训练，只看 reset 后的样子）
python legged_gym/scripts/play.py --task=my_robot
# ③ 小规模训练
python legged_gym/scripts/train.py --task=my_robot --headless --num_envs=512 --max_iterations=200
```

### 8.2 URDF 准备的坑（最常见的失败原因）

| 坑 | 具体表现 | 解决 |
|---|---|---|
| **mesh 路径** | `package://xxx/meshes/foo.STL` → 找不到文件 | 改成**相对 URDF 文件的相对路径**。看 A1 的写法：`../meshes/trunk.dae`（[已验证](resources/robots/a1/urdf/a1.urdf)） |
| **大小写** | URDF 里写 `.STL` 但文件名是 `.stl` | Linux 区分大小写。**Unitree 的 URDF 常犯这个**，用 `grep -o 'filename="[^"]*"'` 逐个核对 |
| **mesh 格式** | Isaac Gym 对某些格式支持不好 | A1 用的是 `.dae`。**优先 `.obj` / `.stl` / `.dae`，具体支持列表见 Isaac Gym 官方文档**（不同版本略有差异） |
| **单位** | 有些 URDF 用 mm 而不是 m | 机器人会大 1000 倍。检查 trunk 的 `<origin>` 和 mesh 尺寸 |
| **固定的虚拟 base** | 有 `base_footprint` 之类的空 link | 用 `collapse_fixed_joints = True`，但**同时要检查 `terminate_after_contacts_on` 不会误匹配** |
| **相机/传感器 link** | URDF 带了 camera、imu 等 link | 会被当成刚体。要么删掉，要么确认它们的质量和碰撞体不影响仿真 |
| **`mimic` 关节** | G1 的手部有 mimic 关节 | **Isaac Gym 不支持 mimic**。训练行走时把手臂关节从 URDF 删掉或固定（很多开源方案就是这么做的） |
| **质量/惯量缺失** | URDF 只有 mesh 没有 `<inertial>` | Isaac Gym 会用密度算，通常不准。真机数据应从 URDF 导出工具或 CAD 拿 |
| **自碰撞** | 人形的手臂会穿进身体 | `self_collisions = 0`（开启）但要小心计算量爆炸；四足一般 `1`（关闭） |

**验证 URDF 的小技巧**：先用 `check_urdf`（ros 工具）或任何 URDF 可视化工具（`urdf-viz`、pybullet 的 `p.connect(p.GUI)` + `loadURDF`）确认模型本身没问题，**再放进 Isaac Gym**。在 Isaac Gym 里调试 URDF 极其痛苦。

### 8.3 四足 vs 人形：参数该怎么定

**这是面试里最可能被追问的地方**——"你只跑过四足，人形有什么不同？"

#### ① PD 增益：人形必须分组

A1 能用 `{'joint': 20.}` 一刀切，因为四足的三个关节规格相近。**人形不行**：髋、膝、踝、肩、肘的扭矩差一个数量级。工业参考值（G1 / H1 量级）：

```python
class control(LeggedRobotCfg.control):
    # ★ 用互不重叠的具体键，靠子串匹配区分关节组
    stiffness = {'hip_yaw': 100., 'hip_roll': 100., 'hip_pitch': 100.,
                 'knee': 150., 'ankle': 40., 'waist': 200.,
                 'shoulder': 40., 'elbow': 40.}
    damping   = {'hip_yaw': 2., 'hip_roll': 2., 'hip_pitch': 2.,
                 'knee': 4., 'ankle': 2., 'waist': 2.,
                 'shoulder': 1., 'elbow': 1.}
    action_scale = 0.25
    decimation = 4        # 50 Hz，和人形多数工作一致
```

**注意两个陷阱**：
- **键必须互不重叠**。`'hip'` 和 `'hip_yaw'` 会同时匹配 `left_hip_yaw_joint`，后者覆盖前者（dict 顺序）。**建议全部用最具体的键。**
- **单位是 N·m/rad**。人形髋关节的 $K_p$ 是四足的 5~8 倍，因为要支撑整个上半身。

#### ② 默认姿态：人形必须"半蹲"

四足是"膝前弓"（`calf = -1.5`）。人形是**膝盖微屈的站立姿态**，绝不能是"腿伸直"：

```python
default_joint_angles = {
    '.*_hip_pitch_joint': -0.1,   # 髋略前倾
    '.*_knee_joint':      0.3,    # ★ 膝盖弯曲（关键！伸直的膝盖无法控制平衡）
    '.*_ankle_pitch_joint': -0.2, # 踝补偿，让脚掌平放
    '.*_shoulder_pitch_joint': 0.0,
    ...
}
```

**为什么膝盖必须弯曲？** 因为膝关节伸直时，腿在运动学上是**奇异位形**：髋和膝的力矩对机身高度的作用退化了，人形会像一根棍子一样倒下去——它没有能力主动调整高度。这是人形和四足最大的结构差异。

**验证方法**：在 `play.py` 里看初始姿态，**膝盖应该是明显弯曲的**，脚掌应该平贴地面。

#### ③ 终止条件：人形宽松得多

```python
terminate_after_contacts_on = ["trunk", "pelvis"]    # 躯干着地才算摔
penalize_contacts_on = ["knee", "elbow", "trunk"]     # 但这些接触会被惩罚
```

**为什么四足可以"机身碰地就算摔"而人形不行？** 因为人形有上肢——摔倒时手会先着地、膝盖会跪地。如果一碰就终止，机器人永远学不会"用手撑一下再站起来"。**所以人形的终止条件要更宽松，靠奖励项去引导"别用手撑地"。**

#### ④ 奖励：人形需要更多项

| 人形额外需要的奖励 | 为什么 |
|---|---|
| `orientation`（**必须打开**） | 人形重心高，姿态失稳是主要失败模式。A1 关掉了这项，人形绝对不能关 |
| `base_height`（**必须打开**） | 人形蹲太深会撞地、站太高会倒。A1 靠间接约束就够了，人形不行 |
| `feet_slip`（脚打滑） | 人形接触相位更复杂 |
| `joint_power` / `energy` | 人形更关心能耗（电池） |
| `feet_contact_forces` | 防止单脚暴力着地 |
| `alive`（存活奖励） | 人形通常需要一项恒定正奖励鼓励"别摔倒" |

**核心差异总结**：

| 维度 | 四足（A1/Go2） | 人形（G1/天工3.0） |
|---|---|---|
| 自由度 | 12 | 23~43 |
| $K_p$ | 20（统一） | 40~200（分组） |
| 平衡难度 | 低（静态稳定，不控制也能站） | **高（静态不稳定，必须动态平衡）** |
| 终止条件 | 机身触地 | 躯干触地 |
| 训练迭代数 | 1000~1500 | 3000~10000+ |
| 显存需求 | 中 | 高（自由度多，状态大） |
| 需要打开的姿态奖励 | 可选 | **必须** |

> **一个务实的建议**：**先做 Go2**。因为 Go2 是四足，能直接复用 A1 的一整套经验，成功率最高、最快出成果。**把 G1/天工 3.0 作为第二步**——当你有了一次完整的四足成功经验，再上人形，成功率会高很多。而且"我先迁移了 Go2 验证流程，然后挑战了 G1 人形"这个叙事，在面试里比"我一上来就搞人形但没搞成"好得多。

### 8.4 ★ 必须重算 `num_observations`（最致命的坑）

**公式**（对照 5.1.4 的表）：

```
num_observations = 9                      # 线速度(3) + 角速度(3) + 投影重力(3)
                 + 3                      # 指令 (vx, vy, ωz)
                 + 3 × num_actions        # 关节角偏差 + 关节速度 + 上一步动作
                 + (187 if measure_heights else 0)
```

**逐个机器人的正确值**：

| 机器人 | DoF | `measure_heights=True` | `measure_heights=False` |
|---|---|---|---|
| A1 / Go2 | 12 | **235** | 48 |
| ANYmal-C | 12 | 235 | 48 |
| **G1**（23 DoF 版本，无手） | 23 | **268** | 81 |
| **G1**（29 DoF 版本，含手） | 29 | **286** | 99 |
| **天工 3.0** | 依版本 | `12 + 3×DoF + 187` | `12 + 3×DoF` |

**怎么验证？** 加一行打印：

```python
# 在 legged_robot.py 的 compute_observations 末尾临时加：
print("实际 obs 维度:", self.obs_buf.shape[-1], " 配置值:", self.cfg.env.num_observations)
```

**或者直接用配置文件覆盖来试**：

```python
class MyRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_actions = 23
        num_observations = 12 + 3*23 + 187      # = 268 ★ 用表达式写，别写死数字
```

**⚠️ 为什么这个错误特别危险？**
- 如果**配置值 > 实际值**：`obs_buf` 是 `(N, 配置值)`，`compute_observations` 里 `self.obs_buf = torch.cat(...)` 会**直接覆盖**成正确维度，然后 `torch.clip` 也是基于新张量……但 `noise_scale_vec` 是按配置值构造的（`_get_noise_scale_vec` 用 `zeros_like(self.obs_buf[0])`），**会导致噪声段错位**，训练能跑但效果差。
- 如果**配置值 < 实际值**：`_get_noise_scale_vec` 建的向量太短，广播时可能报错，或者更糟——**静默地只给前几维加噪**。
- **最隐蔽的情况**：`ActorCritic` 的输入维度用的是 `env.num_obs`（配置值），所以**如果配置值和实际值不一致，网络第一层的输入维度就是错的**，训练必然失败，但报错信息可能很晚才出现。

**结论：改机器人时，`num_actions` 和 `num_observations` 是两个必须同时改、且必须用公式核对的值。**

### 8.5 关于开源资源（务必自行核实）

你的三个目标机器人都有活跃的开源社区，**但我不建议你直接抄——而是理解后再用**：

| 资源方向 | 说明 |
|---|---|
| **宇树官方仿真仓库** | 宇树维护了基于 legged_gym + Isaac Gym 的仓库，覆盖 Go2 / G1 / H1。**搜索关键词：`unitree_rl_gym`**（请自行确认最新地址与版本） |
| **Isaac Lab 的官方任务** | Isaac Lab 内置了 Go2、G1、Anymal 等任务，**这是 2026 年更值得投入的方向** |
| **人形专用框架** | 有若干基于 legged_gym 改的人形版本，加入了"站立/行走/恢复"多阶段奖励。**搜索 `humanoid gym` / `legged_gym humanoid`** |
| **天工（TienKung）** | 天工开源了模型和部分工程，**搜索 `TienKung` / `X-Humanoid`**（具体可用性请自行验证） |

**怎么用这些资源**：
1. **先自己按 8.1 的流程做一遍**（哪怕失败），这样你才知道每一步在干什么
2. 再看别人的实现，**重点对比"他们的 PD 增益、奖励权重、默认姿态和你的有什么不同，为什么"**
3. 把差异记下来——**这些差异就是面试时可以聊的"我踩过的坑"**

---

## 9. 实习面试准备

### 9.1 简历里怎么写这段经历

**不要这么写**（太空，谁都能写）：
> 使用 legged_gym 和 Isaac Gym 训练四足机器人运动控制策略。

**建议这么写**（有数字、有具体动作、有结论）：

> **四足机器人强化学习运动控制（个人项目）**
> - 复现 ETH RSL **legged_gym**（Isaac Gym）框架下的 A1 四足机器人崎岖地形运动策略，4096 并行环境训练 **1500 次迭代（约 1.5 亿仿真步）**
> - 精读并梳理了从 URDF 资产加载 → PD 位置控制 → 235 维观测构造 → 10 项奖励设计 → PPO 训练 → TorchScript 策略导出的**完整流程**，产出一份 5 万字代码精读笔记
> - 定位并修复了上游框架中的一个隐藏缺陷：配置项 `feet_stumble` 无对应的 `_reward_feet_stumble` 实现（因默认值为 0 而未暴露），并用脚本核对了**全部 16 项奖励的配置名与实现名一致性**
> - **扩展**：新增 Go2 / G1 机器人任务定义（配置继承 + 任务注册），重算观测维度、分组整定 PD 增益，完成从四足到人形的参数迁移
> - 熟悉 sim-to-real 技术栈：域随机化（摩擦 64 档分桶 / 质量 / 外力推搡）、观测噪声建模、执行器网络（LSTM）替代理想 PD

**注意最后两条——这就是我建议你接下来要做的**。当前的你如果只跑了 A1 的官方 demo，简历上只能写第一条。**加上"自己迁移了一个新机器人"和"发现了一个 bug"，分量完全不同。**

### 9.2 高频面试题（按被问概率排序）

<details>
<summary><b>Q1: 介绍一下 legged_gym 的整体架构</b></summary>

**答**：三层解耦。
- 底层 **Isaac Gym** 负责 GPU 并行物理仿真，4096 个机器狗同时跑
- 中间 **legged_gym** 把仿真包装成标准 RL 接口（`VecEnv`），定义观测、奖励、终止、地形、域随机化
- 上层 **rsl_rl** 是纯 PPO 实现，完全不知道"四足机器人"这个概念

两层之间只靠 `VecEnv` 这个约 30 行的抽象类通信。这个解耦的好处是我可以把 rsl_rl 换到任何实现 `VecEnv` 的环境上，也可以把 legged_gym 的环境接到任何实现 `VecEnv` 假设的算法上。
</details>

<details>
<summary><b>Q2: 策略的输出是什么？怎么变成关节力矩的？</b></summary>

**答**（★ 必答，最高频）：策略输出 12 维动作，每个动作对应一个关节的**目标位置偏移**：

$$\tau_i = K_p \cdot (\alpha \cdot a_i + q_i^{default} - q_i) - K_d \cdot \dot q_i$$

A1 上 $K_p = 20$，$K_d = 0.5$，`action_scale α = 0.25`。最后力矩会被 clip 到 URDF 里的 `effort` 限幅（髋 20 N·m，大腿/小腿 55 N·m）。

**关键设计思想**：策略不直接输出力矩，而是输出位置目标。这样**低层的稳定控制交给了 PD 控制器，网络只学高层决策**——这是四足 RL 能快速收敛的核心原因。策略只需要学"脚该往哪摆"，不需要学"用多大力"。

**追问：为什么不用直接力矩控制？** 直接力矩（`control_type='T'`）最难训，因为策略要同时学会动力学和运动学。但人形任务有时会用，因为接触相位更复杂、位置控制不够灵活。
</details>

<details>
<summary><b>Q3: 观测里包含什么？为什么这么设计？</b></summary>

**答**：A1 是 235 维，分 8 段：
```
[0:3]   机体系线速度       ×2.0      ← 关键：必须是机体系，不能是世界系
[3:6]   机体系角速度       ×0.25
[6:9]   投影重力           ×1.0      ← 姿态传感器
[9:12]  速度指令           ×[2,2,0.25]
[12:24] 关节角 − 默认姿态   ×1.0      ← 减去 default 让输入零均值
[24:36] 关节速度           ×0.05
[36:48] 上一步动作          ×1.0      ← 让策略知道自己的历史
[48:235] 地形高度 17×11=187  ×5.0      ← 激光雷达的仿真对应物
```

**三个设计要点**：
1. **必须用机体系**——"世界系 +1 m/s"意味着"永远朝东北走"，而任务要的是"朝自己面向的前方走"
2. **所有量都做了缩放**，让量级统一在 ±1 附近，否则关节速度（20 rad/s）会淹没线速度（1 m/s）
3. **加噪声**模拟真机传感器误差，这是 sim2real 的第一根支柱
</details>

<details>
<summary><b>Q4: 奖励函数是怎么设计的？</b></summary>

**答**：由"主奖励 + 正则项"两类构成，A1 实际生效 10 项：

**主奖励**（驱动行为）：
- `tracking_lin_vel = 1.0`：$\exp(-\|v_{cmd} - v\|^2 / 0.25)$
- `tracking_ang_vel = 0.5`：同上，转向
- `feet_air_time = 1.0`：鼓励抬腿迈步

**正则项**（约束不良行为，都是负值）：
- `lin_vel_z = -2.0`：别弹跳
- `ang_vel_xy = -0.05`：别翻滚
- `torques = -0.0002`：省力
- `dof_acc = -2.5e-7`：保护减速器
- `collision = -1.0`：别用腿撞地
- `action_rate = -0.01`：动作平滑
- `dof_pos_limits = -10.0`：别顶死关节

**三个技术细节值得展开**：
1. **用 `exp(-e²/σ)` 而不是直接 $-e^2$**：因为前者有界在 [0,1]，误差大时不会产生巨大梯度把网络带崩；`σ` 控制宽容度。而且上界固定为 1，所以可以用"是否达到 80% 满分"作为命令课程的升级判据。
2. **所有奖励 × `dt`**：把"每步得分"变成"每秒得分率"，这样改控制频率时奖励物理含义不变。
3. **`only_positive_rewards=True`**：总奖励 clip 到 ≥0，让惩罚只能"抵消"奖励而不能制造负奖励，避免早期策略陷入"什么都别做"的退化解。**副作用**是惩罚项如果绝对值过大就会被削平失效，所以调参要看分项日志而不是总数。
</details>

<details>
<summary><b>Q5: 为什么要用 4096 个并行环境？</b></summary>

**答**：因为 **PPO 是 on-policy 算法**——样本用完就扔，所以采样速度直接等于训练速度上限。GPU 并行把采样瓶颈打掉了 2~3 个数量级。

4096 环境 × 24 步 = 每次迭代 9.8 万条经验，1500 次迭代 = 1.5 亿步仿真。这是单环境方案（几天）和这个方案（1~2 小时）的差别——也是论文标题"Learning to Walk in **Minutes**"的由来。

**第二个作用是方差缩减**：4096 只狗在不同地形上各自探索，平均梯度比单只狗的运气可靠得多。

**代价**：显存。光是 rollout buffer 就约 110 MB（24×4096×235×4 字节），加上物理仿真状态，4096 环境大概需要 16~24 GB 显存。
</details>

<details>
<summary><b>Q6: Sim-to-Real 你了解什么？</b></summary>

**答**：三大支柱，legged_gym 里都有对应实现：

1. **观测噪声**（`_get_noise_scale_vec`）：给观测加均匀噪声，噪声幅值参照真机传感器精度（编码器 ~0.01 rad，IMU 姿态 ~0.05 rad）
2. **域随机化**（domain randomization）：与其把仿真做准，不如让策略"见过足够多的不确定性"
   - 摩擦：每环境随机，[0.5, 1.25]，**用 64 档分桶**而不是 4096 个不同值——因为 PhysX 要为每个不同的摩擦组合生成接触材质，分桶能大幅省显存
   - 机身质量：A1 默认关，ANYmal-C 是 ±5 kg
   - **外力推搡**：每 15 秒给机身一个随机速度突变（**不是施加力**——用速度突变模拟冲量更简单可控）
3. **执行器建模**：理想 PD 是"假电机"。ANYmal 的做法是用真机数据训一个 **LSTM 网络**拟合 `(位置误差, 速度) → 实测力矩`。用 LSTM 而非 MLP 是因为电机有迟滞和热效应，输出取决于历史，需要隐藏状态来编码

**另外还有**：随机化初始状态（`_reset_dofs` 里关节角在默认值的 0.5~1.5 倍间随机）——如果每次都从同一姿态开始，策略会过拟合，真机站不起来。
</details>

<details>
<summary><b>Q7: 课程学习是怎么做的？</b></summary>

**答**：这个仓库的地形用"二维编码"：`num_rows=10` 行是难度等级，`num_cols=20` 列是地形类型（斜坡/台阶/障碍物）。**`(i,j)` 一个坐标同时编码了"多难"和"什么类型"**。

课程判据（`_update_terrain_curriculum`）：
- **升级**：一个 episode 内走出了 4 m（地形块长度的一半）→ `terrain_levels += 1`
- **降级**：走得不到"指令速度 × 10 秒"（`max_episode_length_s × 0.5`）→ `terrain_levels -= 1`
- **满级**：回到随机难度，保持探索多样性

**实现上的巧妙之处**：切换难度 = 直接改 `env_origins`（出生点坐标），因为地形是 10×20 的网格，换难度只是换个格子坐标——**不重建任何资源，开销几乎为零**。

还有命令课程（A1 默认关）：如果速度跟踪奖励达到满分的 80%，就扩大指令速度范围（`update_command_curriculum`）。
</details>

<details>
<summary><b>Q8: 训练不收敛你会怎么排查？</b></summary>

**答**：按"从信号到原因"的顺序：

1. **先看 tensorboard**：`mean_reward` 不动？涨一下崩？`episode_length` 很低？
2. **然后看分项奖励 `rew_*`**——这是最关键的技巧。**总奖励涨了不等于学对了**：如果 `rew_lin_vel_z`（惩罚项）也在涨，说明机器人靠疯狂弹跳在骗分。
3. **再查工程性问题**（这些占实际 bug 的绝大多数）：
   - `num_observations` 和实际 obs 维度不匹配（改机器人时最常见）
   - `stiffness` 的键没匹配上任何关节 → $K_p=0$（看日志有没有 "PD gain ... were not defined"）
   - `default_joint_angles` 姿态不合理
   - `terminate_after_contacts_on` **子串匹配**到了不该匹配的 body
4. **最后才调 PPO 超参**：震荡→降学习率/收紧 clip；学得慢→加 `num_steps_per_env`；过早收敛→加 `entropy_coef`

**我实际遇到的例子**：读代码时发现配置项 `feet_stumble` 没有对应的 `_reward_feet_stumble` 实现，只是因为默认值是 0 才没暴露。我用脚本核对了全部奖励的配置名与实现名。**这类"配置与实现静默不一致"的问题是 RL 训练里最难 debug 的一类**——它不报错，只是效果不好。
</details>

<details>
<summary><b>Q9: PPO 的四个关键超参怎么理解？</b></summary>

**答**：
- **`clip_param = 0.2`**：把新旧策略的概率比限制在 [0.8, 1.2]。如果一次更新已经让策略变化太大，就不再给梯度。这是"信任域"的近似实现——比 TRPO 的 KL 硬约束简单得多，这也是 PPO 能统治机器人控制的原因。
- **`gamma = 0.99`**：折扣因子，控制"看得多远"。**注意它和 episode 长度是绑定的**——γ=0.99 时 500 步后的权重只有 0.0066。所以改 `episode_length_s` 时 γ 应该同步调整。
- **`lam = 0.95`**：GAE 的偏差-方差权衡。λ=0 是单步 TD（方差小偏差大），λ=1 是蒙特卡洛（无偏但方差大）。
- **`desired_kl = 0.01` + `schedule='adaptive'`**：这是 rsl_rl 比标准 PPO 多出来的实用改进——每次更新后算新旧策略的 KL，如果 KL > 0.02 就把学习率除以 1.5，如果 KL < 0.005 就乘以 1.5。**所以被问"你的学习率是多少"，标准答案是"初始 1e-3，开启自适应调度在 1e-5~1e-2 之间自动调整"，而不是报一个固定值。**

**另外一个反直觉的点**：PPO 的 `update()` 里对优势做了标准化，所以**把奖励整体乘以 10 倍，训练结果几乎不变**。真正要调的是各项奖励之间的**相对比例**，不是绝对量级。
</details>

<details>
<summary><b>Q10: 时间尺度是怎么安排的？</b></summary>

**答**：三层时间尺度，都在 `_parse_cfg` 里推导出来：

```
物理仿真:  0.005 s (200 Hz)     ← 保证接触求解稳定
   ×4 (decimation)
策略推理:  0.02 s  (50 Hz)      ← 神经网络出一次动作
   ×1000 步
一个 episode: 20 s
```

`self.dt = decimation × sim.dt = 0.02`，然后所有"秒"都基于它换算：`max_episode_length = 20/0.02 = 1000` 步，`push_interval = 15/0.02 = 750` 步。

**为什么要 decimation？** 因为物理需要 200 Hz 才能准确求解接触，但神经网络不需要那么频繁地决策——真机上关节伺服也是 500~1000 Hz 内环 + 50 Hz 策略，这套结构和真机是对得上的。

**`num_steps_per_env = 24` 的选择理由**：24 步 = 0.48 秒 ≈ 1~2 个步态周期。太短则 GAE 自举偏差大，太长则与旧策略差距过大导致 PPO 裁剪失效。
</details>

<details>
<summary><b>Q11: 你的策略网络多大？占多少显存？</b></summary>

**答**：Actor 和 Critic 是**完全独立**的三层 MLP，`[512, 256, 128]`，ELU 激活：

```
Actor:  235 → 512 → 256 → 128 → 12   ≈ 28.7 万参数
Critic: 235 → 512 → 256 → 128 → 1    ≈ 28.5 万参数
```

**训练时总共约 57 万参数，但导出到真机的 JIT 策略只有 Actor 的 28.7 万**（约 1.1 MB）——因为 Critic 在部署时毫无用处。导出时还会 `.to('cpu')` 并 `torch.jit.script` 编译，这样能在 C++ 里直接 `torch::jit::load()`，真机上不需要 Python 环境。

**这个规模小得惊人**——一个 3 层 MLP 学会了机器狗行走。原因是任务结构已经被观测设计和 PD 控制器很好地编码了：策略只需要学"脚往哪摆"这个高层决策，不需要建模动力学。对比 GPT-2 的 15 亿参数，这是完全不同的量级。

**为什么 Actor 和 Critic 不共享底层？** 因为它们的输入可以不同（非对称训练时 Critic 用特权观测），而且共享底层会让两个目标的梯度互相干扰。
</details>

<details>
<summary><b>Q12: 终止条件是怎么设计的？超时算失败吗？</b></summary>

**答**：只有两种终止：
1. **非法接触**：`terminate_after_contacts_on` 里的 body（A1 是 `base` 机身）受到 > 1N 的力
2. **超时**：跑满 1000 步

**关键点：超时不算失败，代码里专门做了处理**（bootstrapping on timeouts）：

```python
if 'time_outs' in infos:
    self.transition.rewards += self.gamma * self.transition.values * infos['time_outs']
```

在超时的那一步，把"未来的价值估计"补回给奖励。**如果不做这件事，机器人会学会"在 19.9 秒时故意摔倒"**——因为反正超时的回报是 0，摔倒也是 0，它没有动力撑到超时。

**另一个容易忽略的点**：`terminate_after_contacts_on` 是**名字子串匹配**，不是精确匹配。所以 URDF 里如果有 `base_link` 和 `base_footprint`，配 `["base"]` 会把两个都匹配上，机器人一碰地就"摔倒"。
</details>

<details>
<summary><b>Q13: 如果让你把策略部署到真机上，要做什么？</b></summary>

**答**：
1. **导出**：`export_policy_as_jit` 已经做了——只 export actor、转 CPU、TorchScript 编译，产出 `policy_1.pt`（约 1.1 MB）
2. **C++ 加载**：`torch::jit::load("policy_1.pt")`，真机部署程序一般是 C++
3. **复现观测**：**这是最容易出错的一步**。真机上必须**逐段复现仿真里的 235 维观测**：
   - 机体系线/角速度 → 状态估计器（仿真直接给真值，真机要用 IMU + 腿式里程计融合，误差会大很多）
   - 投影重力 → IMU 姿态解算
   - **关节顺序必须和 URDF 里的 `<joint>` 顺序一致**（可以用 `env.dof_names` 打印确认）
   - 地形高度 187 维 → 深度相机或激光雷达（**如果真机没有感知，就必须训练盲走（blind）策略，把 `measure_heights` 关掉，obs 变成 48 维**）
4. **复现动作→力矩的映射**：`τ = K_p(0.25a + q_default − q) − K_d·q̇`，$K_p/K_d$/`action_scale` **必须和训练时完全一致**
5. **控制频率对齐**：策略 50 Hz，关节伺服 500~1000 Hz，中间要做插值（真机上一般用线性插值或零阶保持）
6. **安全检查**：力矩限幅、关节限位、遥控器急停

**最容易被低估的是第 3 步**——如果真机没有地形感知，那 187 维的高度信息就没有来源。**这也是为什么很多真机部署用的是盲目（blind）策略**：把 `terrain.measure_heights = False`，obs 维度变成 48，训练一个只靠本体感觉的策略。它的鲁棒性差一些，但部署简单得多。
</details>

<details>
<summary><b>Q14: 你训练的时候用了多久？有没有遇到什么问题？</b></summary>

**答**（★ 这题是给你铺路的，讲你的真实经历）：
> 我用 A1 在崎岖地形任务上跑满了 1500 次迭代，4096 个并行环境，约 1.5 亿仿真步。训练过程中主要的时间花在读代码和理解各个模块上，而不是训练本身。
>
> 我遇到的最有意思的一个问题是：在读奖励函数的时候，我注意到配置里有 `feet_stumble` 这一项，但代码里实现的方法叫 `_reward_stumble`。因为 `_prepare_reward_function` 是通过字符串拼接 `'_reward_' + name` 然后用 `getattr` 反射取方法的，所以这个不一致会导致 `AttributeError`——只是因为它的默认值是 0，在准备阶段就被 `pop` 掉了，所以从来没暴露。我写了个脚本核对了全部 16 项奖励的配置名和实现名，确认只有这一处。
>
> 这件事让我意识到，这个框架里"配置写错了不会报错、只会静默失效"是一个系统性的风险点，所以后来我做参数迁移的时候都会先跑小规模测试确认。

**这个回答的价值**：它展示了你①真的读了代码②有验证意识③能从具体问题上升到系统性认识。
</details>

<details>
<summary><b>Q15: 你了解 Isaac Lab 吗？为什么还要用 Isaac Gym？</b></summary>

**答**（★ 这题现在很可能会问，因为 Isaac Gym 已经停止维护）：
> 了解。Isaac Gym 是 NVIDIA 的上一代 GPU 并行仿真方案，2024 年 NVIDIA 把重心转向了 Isaac Sim / Isaac Lab，legged_gym 的 README 里也明确建议迁移。
>
> 我这套代码是基于 legged_gym 的，主要原因是**它是最经典的四足 RL 教学材料**——ETHz 的这篇 "Learning to Walk in Minutes" 是这个领域的奠基性工作，代码结构清晰、注解完善，适合把原理吃透。
>
> 而且**关键概念是通用的**：观测构造、PD 位置控制、奖励设计、域随机化、地形课程、PPO 超参——这些在 Isaac Lab 里的设计思路完全一致，只是 API 变了。我先用 legged_gym 建立概念框架，接下来计划把同样的任务迁移到 Isaac Lab 上验证。

**这个回答展示了"我知道技术迭代，但我知道该先学什么"的判断力。**
</details>

### 9.3 可能被追问的"深水区"

如果面试官很懂行，可能会追问：

| 问题 | 要点 |
|---|---|
| **187 维高度是怎么测的？** | **不是射线检测**，是地形高度图查表：采样点投影到栅格索引，取 3 个角点的 **min**（保守估计） |
| **为什么摩擦随机化只用 64 档？** | PhysX 为每个不同摩擦组合生成接触材质，分桶省显存 |
| **`only_positive_rewards` 有什么副作用？** | 惩罚项被"削平"失效——绝对值超过正奖励总和的惩罚实际上不起作用，所以要看分项日志 |
| **`action_rate` 和 `dof_acc` 有什么区别？** | 前者惩罚**动作（目标位置）变化**，后者惩罚**关节加速度**。前者约束策略输出平滑度，后者约束物理运动的剧烈程度。两个都要，因为它们约束的对象不同 |
| **为什么 `feet_air_time` 里要用 `contact OR last_contacts`？** | PhysX 在三角网格上的接触报告会闪烁，用"或"做个滤波器 |
| **`heading_command` 是干什么的？** | 上层给目标朝向，代码用 P 控制器（增益 0.5）算出需要的 ωz 指令。为了让仿真接口和真机导航接口一致 |
| **`self_collisions = 1` 是开启还是关闭？** | **关闭**（1 是位掩码，表示禁用自碰撞的位） |
| **为什么导出策略时 `.to('cpu')`？** | 真机（Jetson/工控机）通常没有 CUDA 或不希望占用 GPU |

### 9.4 面试前的自查清单

- [ ] 能**手画出**数据流图：obs → 策略 → action → PD → 力矩 → 物理 → 奖励/终止/新 obs
- [ ] 能**默写** A1 的 235 维观测分段表
- [ ] 能**写出** PD 力矩公式并解释每一项
- [ ] 能说出 A1 的 **10 个生效奖励**和各自的作用
- [ ] 能解释 **decimation 和时间尺度**的关系
- [ ] 能说清 **PPO 的 clip 和自适应 KL** 在代码的哪几行
- [ ] 能讲出 **sim2real 三大支柱**并对应到代码位置
- [ ] 知道自己训练用的**具体数字**（1500 迭代 / 4096 环境 / 1.5 亿步）
- [ ] 准备好一个**踩坑故事**（`feet_stumble` 那个）
- [ ] 能说出至少**一个改进想法**（见下）

### 9.5 一个能让你脱颖而出的"改进想法"

如果面试官问"你觉得这个框架有什么可以改进的地方"，**不要泛泛而谈**。给一个具体的、你验证过的：

**想法 1（最容易做）**：**修复 `_reward_feet_stumble` 命名不一致**，然后**加一个自动化检查**——在 `_prepare_reward_function` 里检查 `hasattr`，如果找不到方法就抛出清晰的错误信息，而不是等到运行时才崩。这是"从 bug 到防止同类 bug 的机制"，比单纯修 bug 高一个层次。

**想法 2（有技术含量）**：**扩充 `terrain_proportions` 启用落石/沟/坑地形**。因为 `make_terrain` 里有 8 个分支，但默认 5 个比例只够走到第 5 个，后面 3 种（stepping stones / gap / pit）是**死代码**。把这个开启并验证 A1 能否学会过沟。**这既展示了你读代码的深度，又能做出可视化的成果（机器人过沟的视频）。**

**想法 3（有研究价值）**：**给 A1 加执行器网络**。ANYmal 有而 A1 没有。做法是用 `control_type='T'` 先训一个"已知真机模型"的 PD 数据，或者（更实际）**在 PD 输出后加一层"电机延迟 + 力矩噪声 + 死区"** 来近似真机特性，然后对比有无这层建模的训练效果差异。**这是能直接写进简历的完整实验。**

---

## 10. 四周学习路线

假设你每天能投入 3~4 小时：

### 第 1 周：把框架跑通 + 建立概念

| 天 | 任务 | 产出 |
|---|---|---|
| 1 | 装环境（7.1）；跑通 `anymal_c_flat`（最快，300 迭代，20 分钟） | 能看到会走路的 ANYmal |
| 2 | 读第 2 章（RL 概念）+ 第 4 章（训练时间线） | 能画出数据流图 |
| 3 | 跑通 A1 崎岖地形训练；学会看 tensorboard | 能解读训练曲线的每一项 |
| 4-5 | 精读 `legged_robot.py` 的 `step` / `post_physics_step` / `compute_observations` / `_compute_torques` | 在 5.1.2~5.1.4 上做自己的批注 |
| 6 | 精读奖励机制（5.1.5~5.1.7）；**动手验证 `feet_stumble` bug** | 能列出十项奖励的作用 |
| 7 | 精读 rsl_rl 的 `on_policy_runner.learn` + `ppo.update` | 能讲清 PPO 的四个损失项 |

### 第 2 周：动手调参 + 认坑

| 天 | 任务 | 产出 |
|---|---|---|
| 1 | 做**单变量实验**：把 `action_scale` 从 0.25 改成 0.1 和 0.5，各跑 200 迭代 | 一组对比曲线 |
| 2 | 同上，改 `stiffness` / `damping` | 知道什么"站不住" |
| 3 | 同上，改 `tracking_sigma`（0.1 / 0.25 / 0.5） | 理解收益函数的形状 |
| 4 | 关掉 `measure_heights`，看 obs 变 48 维后能不能学会（盲走） | 盲走策略 |
| 5 | 打开 `terrain.curriculum = False`，看纯随机地形下的表现 | 理解课程学习的价值 |
| 6 | 打开 `feet_stumble = -0.1`，**亲手复现那个 AttributeError** | 一段可以写进笔记的 debug 记录 |
| 7 | 整理**实验记录表**（参数 / 结果 / 结论） | 你的第一个实验报告 |

> **实验方法论很重要**：**每次只改一个参数**，其他全固定，跑同样的迭代数，对比曲线。**不要一次改五个参数然后猜哪个起作用。** 这个习惯在实习面试里也是加分项——面试官会问"你怎么确定是那个参数起的作用"。

### 第 3 周：迁移一个新机器人

| 天 | 任务 | 产出 |
|---|---|---|
| 1 | 准备 Go2（或 A1 变体）的 URDF，验证模型本身没问题 | 能在别的可视化工具里打开 |
| 2 | 写配置（8.1 四步）；重算 `num_observations`（8.4） | 能通过 `test_env.py` |
| 3 | 用 `play.py` 看初始姿态，调 `default_joint_angles` | 机器人能"站"起来 |
| 4 | 小规模训练 512 环境 × 200 迭代，调 PD 增益 | 能走几步 |
| 5-6 | 全规模训练 + 调奖励 | 完整的行走视频 |
| 7 | 写实验记录 + 录制视频 | **简历素材** |

### 第 4 周：做人形 + 准备面试

| 天 | 任务 | 产出 |
|---|---|---|
| 1-2 | 把同样的流程套用到 G1（重点：分组 PD 增益 + 打开 orientation/base_height 奖励） | 至少跑通环境 |
| 3 | 完成一个"改进实验"（9.5 里的想法 1 或 2） | 可展示的成果 |
| 4-5 | 准备面试题（9.2），对着镜子讲一遍 | 能流畅讲 5 分钟 |
| 6 | 写简历（9.1） | 一页纸 |
| 7 | 把整个过程的笔记整理成博客/README | 面试时可展示的材料 |

---

## 11. 附录

### 11.1 术语中英对照

| 英文 | 中文 | 在本仓库的体现 |
|---|---|---|
| Observation / Obs | 观测 | `obs_buf`，A1 为 235 维 |
| Action | 动作 | `actions`，12 维，被 clip 到 ±100 |
| Reward | 奖励 | `rew_buf`，10 项加权求和 |
| Policy | 策略 | `ActorCritic.actor` |
| Value Function | 价值函数 | `ActorCritic.critic` |
| Advantage | 优势函数 | `storage.advantages` |
| GAE (Generalized Advantage Estimation) | 广义优势估计 | `compute_returns` |
| Episode / Rollout | 回合 / 采样轮 | `episode_length_buf` / `num_steps_per_env` |
| Termination | 终止 | `check_termination` |
| Decimation | 抽稀 / 分频 | `control.decimation = 4` |
| Domain Randomization | 域随机化 | `cfg.domain_rand` |
| Curriculum Learning | 课程学习 | `_update_terrain_curriculum` |
| Privileged Observation | 特权观测 | `num_privileged_obs` |
| Sim-to-Real | 仿真到真机迁移 | 噪声 + 域随机化 + 执行器网络 |
| Bootstrapping | 自举 | 超时时的价值回填 |
| Actor-Critic | 演员-评论家 | `ActorCritic` 类 |
| On-policy | 同策略 | PPO 的特性 |
| Stiffness / Damping | 刚度 / 阻尼 | $K_p$ / $K_d$ |

### 11.2 关键代码行速查（打印贴在显示器上）

| 想改什么 | 去哪一行 |
|---|---|
| 力矩怎么算 | [legged_robot.py:368](legged_gym/envs/base/legged_robot.py#L368) |
| 观测有哪些 | [legged_robot.py:212-219](legged_gym/envs/base/legged_robot.py#L212-L219) |
| 什么时候摔倒 | [legged_robot.py:141](legged_gym/envs/base/legged_robot.py#L141) |
| 奖励怎么加权 | [legged_robot.py:196-200](legged_gym/envs/base/legged_robot.py#L196-L200) |
| 奖励名怎么变函数 | [legged_robot.py:560-563](legged_gym/envs/base/legged_robot.py#L560-L563) |
| 时间尺度 | [legged_robot.py:729](legged_gym/envs/base/legged_robot.py#L729) |
| 摩擦随机化 | [legged_robot.py:266-277](legged_gym/envs/base/legged_robot.py#L266-L277) |
| 地形难度分级 | [terrain.py:85-92](legged_gym/utils/terrain.py#L85-L92) |
| 地形类型选择 | [terrain.py:122-143](legged_gym/utils/terrain.py#L122-L143) |
| 训练主循环 | [on_policy_runner.py:102-142](rsl_rl/rsl_rl/runners/on_policy_runner.py#L102-L142) |
| PPO 损失 | [ppo.py:154-171](rsl_rl/rsl_rl/algorithms/ppo.py#L154-L171) |
| 自适应学习率 | [ppo.py:139-151](rsl_rl/rsl_rl/algorithms/ppo.py#L139-L151) |
| GAE | [rollout_storage.py:123-137](rsl_rl/rsl_rl/storage/rollout_storage.py#L123-L137) |
| 策略采样 vs 推理 | [actor_critic.py:123-132](rsl_rl/rsl_rl/modules/actor_critic.py#L123-L132) |
| 策略导出 | [helpers.py:180-190](legged_gym/utils/helpers.py#L180-L190) |

### 11.3 你当前仓库的状态（我读代码时看到的）

| 项 | 状态 |
|---|---|
| `rsl_rl/` | 已内置（v1.0.2 结构），无需再 clone |
| 训练记录 | `logs/rough_a1/Sep21_13-29-25_/`，**已跑满 1500 迭代**（model_0 ~ model_1500，每 50 存一次） |
| 导出策略 | `logs/rough_a1/exported/policies/policy_1.pt` 已存在（≈1.1 MB，可直接 C++ 加载） |
| 代码改动 | [legged_robot.py:266](legged_gym/envs/base/legged_robot.py#L266) 加了一行中文注释（`#域随机化`） |

> **建议**：把 `logs/rough_a1/Sep21_13-29-25_/` 备份一份，然后重命名成有意义的名字（比如 `logs/rough_a1/baseline_1500iter/`）。**实验管理习惯**（每次实验记录改了哪些参数）在实习里很重要——面试官问"你做过什么实验"时，你能直接翻出记录，比"我试过一些参数"强太多。

### 11.4 参考资源

| 资源 | 用途 | 备注 |
|---|---|---|
| 论文 *Learning to Walk in Minutes Using Massively Parallel Deep RL* (arXiv:2109.11978) | 理解设计动机 | 代码里的每个设计决策论文里都有解释 |
| 本站 [README.md](README.md) | 官方说明 + 新增机器人的官方流程 | 短小但关键 |
| Isaac Gym 官方文档（下载包里的 `docs/index.html`） | API 查询、**mesh 格式支持列表** | 必查 |
| 本仓库 [legged_robot.py](legged_gym/envs/base/legged_robot.py) | 一切的核心 | 906 行，值得读三遍 |
| rsl_rl 官方仓库 | 对比新旧版本差异 | ⚠️ 注意本仓库需要 v1.0.2 |
| Isaac Lab 官方文档 | 下一步方向 | 2026 年更值得投入 |

---

## 最后：给你的三句话

1. **这个仓库不难，难在"不知道从哪看起"。** 你现在有了第 5.1.1 节那张行号表，按图索骥就行——**200 行核心代码撑起了整个系统**。

2. **跑通 ≠ 学会，改动 ≠ 理解。** 真正让面试官记住你的是"**我改了这个参数，观察到了这个现象，我推断原因是这个**"这样的叙事。所以第 2 周的单变量实验比第 1 周的跑通更重要。

3. **你现在最该做的下一步（如果只做一件事）**：**用 `play.py` 打开你的 `rough_a1` 策略，看着它走路，然后问自己"它是怎么知道该抬哪条腿的"。** 然后回到 `compute_observations` 和 `reward_tracking_lin_vel`，你会突然明白——**这就是自学的正确姿势：先建立直觉，再用代码解释直觉。**
