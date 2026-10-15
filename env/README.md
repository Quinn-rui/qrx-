# 环境配置记录

这里保存各个 conda 环境的"配方"，用于**换电脑或重装时一键还原**，以及记录环境里到底装了什么。

## 文件说明

| 文件 | 内容 | 作用 |
|------|------|------|
| `*.environment.yml` | conda 管理的包 | 定义 Python 版本和系统级依赖 |
| `*.requirements.txt` | pip 管理的包 | 定义 Python 库及其版本 |

> ⚠️ **注意 `-e`（源码安装）的包**：`conda env export` 记录不了它们，
> 所以 isaacgym / legged_gym / rsl_rl 这三个不在 `environment.yml` 里，
> 只在 `requirements.txt` 末尾以注释形式列出了手动安装步骤，必须手动装。

## 两个文件怎么选

| | `environment.yml`（conda env export） | `requirements.txt`（pip freeze） |
|---|---|---|
| conda 装的包 | ✅ 记录 | ❌ 输出本机临时路径，不可用 |
| pip 装的包 | ✅ 记录（在 `pip:` 子区） | ✅ 记录 |
| `-e` 源码装 | ❌ 漏掉 | ⚠️ 记录了但不能直接使用 |

**优先用 `conda env export`**，它两种包都能记，只有 `-e` 需要手动补。
`pip freeze` 只适用于「该环境的包全部由 pip 安装」的情况。

## 还原步骤（以 leggedGym 为例）

```bash
# 1. 用 environment.yml 建环境（会自动装 Python 和 pip 的包）
conda env create -f env/leggedGym.environment.yml

# 2. 激活
conda activate leggedGym

# 3. 装那三个需要手动处理的包（文件末尾有说明）
#    a) Isaac Gym：解压 IsaacGym_Preview_4_Package 后
cd <IsaacGym_Preview_4_Package>/isaacgym/python && pip install -e .

#    b) legged_gym（含 rsl_rl 子模块）
git clone https://github.com/leggedrobotics/legged_gym.git
cd legged_gym && git submodule update --init
pip install -e .

#    c) rsl_rl
cd rsl_rl && pip install -e .

# 4. 补齐剩余的 pip 包
pip install -r env/leggedGym.requirements.txt
```

## 环境一览

| 环境名 | Python | 主要用途 | 关键包 |
|--------|--------|---------|--------|
| `leggedGym` | 3.8 | 四足机器人强化学习训练 | Isaac Gym, legged_gym, rsl_rl, torch 2.4.1+cu121 |
| `embodied` | 3.8 | 具身智能 / MuJoCo 相关 | mujoco 3.2.3 |
| `base` | — | conda 本体，**保持干净，不要往里装包** | — |

## 小提醒

- `-e`（editable）安装的包，**源代码文件夹不能移动位置**，否则 `import` 会失败。
- 新建空环境很便宜，但重新装 torch + CUDA 很贵。**能克隆就克隆**：
  ```bash
  conda create -n 新环境 --clone leggedGym
  ```
