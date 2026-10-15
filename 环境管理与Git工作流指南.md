# 环境管理与 Git 工作流指南

> 面向：刚开始接触机器人仿真 / 具身智能，被 Python 环境搞晕过、对 Linux 不熟、想把自己的工作管理起来的人。
>
> 这份文档解决三个问题：
> 1. **环境管理** —— 为什么会有"装了却找不到"、怎么管、新项目怎么办
> 2. **Git 日常工作流** —— 怎么用 git 记录自己每天干了什么
> 3. **GitHub 仓库维护** —— 怎么把一个仓库慢慢养好

---

## 目录

- [第一部分：原理 —— 为什么需要环境隔离](#第一部分原理--为什么需要环境隔离)
- [第二部分：日常环境管理](#第二部分日常环境管理)
- [第三部分：做一个新项目的完整流程](#第三部分做一个新项目的完整流程)
- [第四部分：依赖记录与还原](#第四部分依赖记录与还原)
- [第五部分：Git —— 记录你每天干了什么](#第五部分git--记录你每天干了什么)
- [第六部分：GitHub —— 逐步完善你的仓库](#第六部分github--逐步完善你的仓库)
- [第七部分：排错速查（FAQ）](#第七部分排错速查faq)
- [附录：命令速查表](#附录命令速查表)

---

# 第一部分：原理 —— 为什么需要环境隔离

## 1.1 从一个真实的困惑说起

> "我之前用 Isaac Gym 训练的时候装了 numpy，昨天用 MuJoCo 跑一个 python 文件却显示没有 numpy。这个环境隔离我不懂。"

这是几乎每个人都会踩的第一个坑。要理解它，得先知道 **Python 是怎么找包的**。

## 1.2 `import` 到底发生了什么

当你写：

```python
import numpy
```

Python 不会"在电脑里到处找"，它**只会在当前那个 Python 的搜索路径里找**。这个路径可以用一条命令看到：

```bash
python -c "import sys; print(sys.path)"
# 或者更直观的：
which python
```

`which python` 会告诉你"现在用的是哪个 Python"。**这个 Python 能看见哪些包，完全取决于它是谁。**

## 1.3 环境 = 磁盘上一个独立的文件夹

一个 conda 环境，本质上就是磁盘上的一个目录：

```
miniconda3/                        ← base 环境
miniconda3/envs/embodied/          ← 一个环境
miniconda3/envs/leggedGym/         ← 另一个环境
miniconda3/envs/robot_base/        ← 再一个环境
```

每个文件夹里都有一套**独立的**：

```
envs/leggedGym/
├── bin/python          ← 独立的 Python 解释器
├── bin/pip             ← 独立的 pip
├── lib/python3.8/site-packages/   ← 独立的一堆包
│   ├── numpy/
│   ├── torch/
│   └── ...
└── ...
```

**关键结论：**

> `import numpy` 时，Python 只会去"当前激活的那个环境的 `site-packages`"里找。
> 别的环境里有没有 numpy，跟它一点关系都没有。

所以"Isaac Gym 环境有 numpy，MuJoCo 环境没有"——因为它们是**两个文件夹**，numpy 只装在其中一个里面。

## 1.4 为什么必须隔离？不隔离不行吗？

不行，因为**版本会打架**。

举个真实例子：

| 项目 | 需要的 numpy | 原因 |
|------|-------------|------|
| legged_gym (Isaac Gym) | `1.23.5` | Isaac Gym 的 C++ 扩展是按老版本 ABI 编译的 |
| 新的具身智能项目 | `2.x` | 新库普遍要求 numpy 2 |

**numpy 只能有一个版本存在于同一个文件夹里。** 如果全装在一起，装了 2.x，Isaac Gym 就崩；装回 1.23，新项目就跑不了。

环境隔离就是：**让每个项目有自己的文件夹，各装各的版本，互不干扰。**

## 1.5 conda 和 pip 的分工

这两个工具经常一起出现，容易搞混。简单区分：

| | conda | pip |
|---|-------|-----|
| 装什么 | 任何东西，包括**非 Python 的**（C 库、CUDA、编译器） | 只能装 **Python 包** |
| 从哪装 | conda 仓库（anaconda / conda-forge） | PyPI |
| 典型用途 | 装 Python 本体、numpy/scipy/torch 这类**有 C 扩展的** | 装纯 Python 的库，或 GitHub 上的源码 |
| 命令 | `conda install xxx` / `conda create` | `pip install xxx` |

**实践中常见的做法：**

```bash
conda create -n 环境名 python=3.10    # Python 本体用 conda
conda install numpy pytorch             # 重家伙用 conda（或 pip，都行）
pip install 一些小的纯 Python 包        # 其余的用 pip
```

> ⚠️ 但要注意：**谁装的，就要用谁去记录/卸载**。混着装之后，`pip freeze` 只能记录 pip 装的那部分（后面第四部分会详细讲这个坑）。

## 1.6 还有一层：不只是 conda，还有 path 和 LD_LIBRARY_PATH

`import` 找的是 **Python 包**。但有些包内部还要加载 **C/C++ 动态库**（`.so` 文件），这是操作系统层面的事，由 `LD_LIBRARY_PATH` 这个环境变量控制。

真实案例（就是本项目踩过的坑）：

```
ImportError: libpython3.8.so.1.0: cannot open shared object file
```

这不是"包没装"，而是"包装了，但它依赖的 `.so` 文件操作系统找不到"。解决办法是让 `LD_LIBRARY_PATH` 包含环境的 `lib` 目录。

**持久化的正确做法**：在环境的 `activate.d` 目录里放个脚本，每次 `conda activate` 自动执行：

```bash
# 文件：envs/环境名/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

> `$CONDA_PREFIX` 是 conda 的变量，指向"当前激活环境的目录"，所以这句话对所有环境都通用，不用写死路径。

## 1.7 一句话总结原理

```
环境 = 一个独立文件夹
     = 一套独立的 Python + 一堆独立的包

import 只会在"当前激活环境"的文件夹里找
所以：不同环境之间互不影响 —— 这是特性，不是 bug
```

---

# 第二部分：日常环境管理

## 2.1 三条最重要的命令

**这三条命令能解决 80% 的"环境问题"。**

```bash
conda env list      # ① 我有哪些环境？
which python        # ② 我现在在哪个环境？ ← 最重要！
pip list            # ③ 当前环境装了些什么？
```

### `which python` 是你的救命命令

任何时候遇到"明明装了却报找不到"，**第一件事就是敲 `which python`**。

正常输出（在 leggedGym 环境里）：

```
~/miniconda3/envs/leggedGym/bin/python
              ↑ 这里显示的是当前环境名
```

如果输出是 `~/miniconda3/bin/python`，说明你在 **base** 环境，而不是你以为的那个环境——**这就是绝大多数"包找不到"的真正原因**。

## 2.2 你的环境清单

| 环境 | Python | 用途 | 关键内容 |
|------|--------|------|---------|
| `base` | — | **conda 本体** | ⚠️ 保持干净，永远不要往里装项目包 |
| `embodied` | 3.8 | 具身 / MuJoCo 相关 | mujoco |
| `leggedGym` | 3.8 | 四足机器人强化学习 | Isaac Gym、legged_gym、rsl_rl、torch 2.4.1+cu121 |
| `robot_base` | 3.10 | **母版**，新项目的起点 | numpy、scipy、matplotlib、ipython |

> **为什么 base 要保持干净？**
> 因为 base 是 conda 自己的家。往里装项目包，轻则版本冲突让 conda 自己出问题，重则整个 conda 崩掉。
> 记住口诀：**先 `conda activate`，再 `pip install`。**

## 2.3 每天开工的标准动作

```bash
conda activate 项目环境    # ① 进入要用的环境
which python               # ② 确认进对了（别跳过这步）
cd 项目文件夹               # ③ 进入项目目录
# ...开始干活...
```

## 2.4 ⭐ 判断：什么时候才需要新建环境？

**这是新手最容易搞错的地方。**

> ❌ 错误认知：每来一个新项目，就要新建一个环境，然后重新装一大堆东西。
>
> ✅ 正确认知：**只有在"依赖冲突"时才需要新建环境。**

判断表：

| 情况 | 要新建环境吗 | 怎么做 |
|------|------------|-------|
| 新项目要的包，现有环境都有、版本也兼容 | ❌ 不用 | 直接用现有环境 |
| 新项目要 numpy 2.x，现有环境是 1.23 | ✅ 要 | 两个版本的 numpy 无法共存 |
| 新项目要 Python 3.11，现有环境是 3.8 | ✅ 要 | 一个环境只能有一个 Python 版本 |
| 新项目要 torch 2.4，现环境是 torch 1.10 | ✅ 要 | 大版本冲突 |
| 新项目就是另一个四足机器人训练 | ❌ 不用 | 用 `leggedGym`，或克隆一个 |

## 2.5 三种复用方式（避免"从零配一大堆"）

### 方式 1：克隆（最省事，推荐日常使用）

```bash
conda create -n 新环境名 --clone robot_base
```

**本质是本地复制文件夹**，几秒钟完成。适合"新项目和现有环境需求差不多"的情况。

实测：克隆一个含 numpy/scipy/matplotlib 的环境只需 **5 秒**；而从零装这些要好几分钟。

### 方式 2：按配方重建（跨机器 / 需要精确还原）

```bash
conda env create -f environment.yml
```

适合换电脑、或者在另一台机器上复现环境。配方哪来的见第四部分。

### 方式 3：从零建 + 只装增量（最干净）

```bash
conda create -n 新环境名 python=3.10
conda activate 新环境名
pip install 只有这个项目要的包
```

适合确定知道要什么的情况。

## 2.6 环境的分层心智模型

理解这个模型，你就不会再觉得"每次要从零配一大堆"：

```
层级              例子                            是否可复用
──────────────────────────────────────────────────────────
① Python 本体     python 3.10                     可以，随手就有
② 科学计算底座    numpy / scipy / matplotlib      可以，几乎所有项目都要
③ 深度学习底座    torch + CUDA ← 最难装的          可以，同方向项目通用
④ 项目专属        某个具体库 / 自己写的代码        每个项目不同
```

**关键洞察：**

- 真正难装的只有 ③（torch + CUDA），而它是**同方向项目通用的**
- 一个环境里可能装了 60+ 个包，但真正"属于这个项目"的往往只有几个
- 剩下的都是共享的底座，**不需要重新装**

**所以"母版环境"（`robot_base`）的价值就在这**：把 ①② 层一次性装好，以后所有新项目都从它克隆，只装 ④ 层的增量。

---

# 第三部分：做一个新项目的完整流程

这一部分是**可以照着敲的操作手册**。

## 场景

你要开始一个新项目，叫 `my_project`。

## Step 1：判断要不要新环境

对照 [2.4 的判断表](#24--判断什么时候才需要新建环境)：

- 需求和老环境兼容 → 直接用老环境，跳到 Step 3
- 有冲突 → 继续 Step 2

## Step 2：从母版克隆环境

```bash
conda create -n my_project --clone robot_base
```

## Step 3：激活并确认

```bash
conda activate my_project
which python          # 必须显示 envs/my_project/bin/python
```

## Step 4：建项目文件夹

```bash
mkdir -p ~/projects/my_project
cd ~/projects/my_project
```

> 建议把项目都放在一个统一的地方（比如 `~/projects/`），别散落在桌面、下载、文档各处。
> 项目文件夹和环境是**两个概念**：环境在 `miniconda3/envs/`，项目代码在 `~/projects/`。

## Step 5：初始化 Git（可选，但强烈建议）

```bash
git init
```

## Step 6：装这个项目额外需要的包

```bash
pip install 包名
# 或
conda install 包名
```

**这一步只装"增量的那一两个"**，不要重新装 numpy/torch 这些底座。

## Step 7：写 `.gitignore`（Git 项目必做）

建一个 `.gitignore` 文件，告诉 git 哪些东西不用管：

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# 训练产物（很大，别传）
logs/
wandb/
outputs/
*.pt
*.pth
*.ckpt

# 编辑器 / 系统
.vscode/
.idea/
.DS_Store
```

## Step 8：记录依赖配方

```bash
conda env export | grep -v "^prefix:" > environment.yml
```

## Step 9：第一次提交

```bash
git add .
git commit -m "init: 项目初始化"
```

## Step 10：关联远程仓库并推送

```bash
# 先在 GitHub 网页上建一个空仓库，然后：
git remote add origin git@github.com:你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

> `-u` 只需第一次加。之后直接 `git push` 就行。

## 完整流程速览

```
判断要不要新环境
   ↓ 要
克隆母版  →  激活  →  建项目文件夹  →  git init
   ↓
装增量包  →  写代码  →  写 .gitignore  →  记录 environment.yml
   ↓
git add / commit / push
```

---

# 第四部分：依赖记录与还原

## 4.1 为什么需要"配方"

环境本身在 `miniconda3/envs/` 里，**不在你的项目仓库里**。所以：

- 换电脑 → 环境不会跟着走
- 重装系统 → 环境全没了
- 想分享给别人 → 别人不知道怎么配

**"配方"就是一份文字记录**，写了"这个环境需要哪些包、什么版本"，让你能一键重建。

## 4.2 ⭐ 两个工具的坑（这一节很重要）

### `conda env export` 与 `pip freeze` 的差异

| | `conda env export` | `pip freeze` |
|---|---|---|
| conda 装的包 | ✅ 正确记录 | ❌ **输出本机临时路径，完全不可用** |
| pip 装的包 | ✅ 正确记录（在 `pip:` 子区） | ✅ 正确记录 |
| `pip install -e .` 源码装的包 | ❌ 漏掉 | ⚠️ 记录了，但格式不能直接用 |

### 坑 1：pip freeze 遇到 conda 包会输出垃圾

如果一个包是 conda 装的，`pip freeze` 会输出成这样：

```
numpy @ file:///home/task_177619032023410/croot/numpy_and_numpy_base_.../work/dist/numpy-2.2.5-...whl
```

那些 `file:///home/task_...` 是 **conda 构建包的服务器上的临时路径**，在你的电脑上根本不存在。用这份清单去装包，必然失败。

**只有当一个环境的包全部由 pip 安装时，`pip freeze` 才是可靠的。**

### 坑 2：`-e` 源码安装的包，两个工具都记不好

```bash
pip install -e .    # -e = editable，可编辑安装
```

这种安装**不会复制一份代码到环境里**，而是记下一个**指向源代码文件夹的路径**。所以：

- `conda env export`：**完全漏掉**（它不认识这种）
- `pip freeze`：记录了，但是 `-e /本机/绝对/路径` 的格式，换台机器就失效

**正确做法**：在文档里**手动写明安装步骤**。例如：

```
# 需手动安装（源码 + -e）：
#   1. git clone https://github.com/xxx/yyy.git
#   2. cd yyy && pip install -e .
```

> 这就是为什么 `env/leggedGym.requirements.txt` 末尾有那三条"需手动"的注释。

### 结论：优先用 `conda env export`

```bash
conda env export | grep -v "^prefix:" > environment.yml
```

- `grep -v "^prefix:"` 是为了**去掉 `prefix:` 那行**——它写着本机的绝对路径，换机器会出错
- 这一条命令能用的情况最广，conda 包和 pip 包都能记
- 唯一例外是 `-e` 源码包，需要手动补说明

## 4.3 怎么还原环境

```bash
# 1. 按配方建环境（pip 装的包也会一起装上）
conda env create -f environment.yml

# 2. 激活
conda activate 环境名

# 3. 手动安装那些 -e 源码包（按文档里的说明）
#    例如：
#    cd <IsaacGym_Preview_4_Package>/isaacgym/python && pip install -e .
#    git clone https://github.com/leggedrobotics/legged_gym.git
#    cd legged_gym && pip install -e .
#    cd rsl_rl && pip install -e .

# 4. 验证
python -c "import 关键包; print('OK')"
```

## 4.4 把配方放进仓库

推荐在仓库里开一个 `env/` 目录：

```
你的仓库/
└── env/
    ├── README.md                       ← 说明怎么还原
    ├── leggedGym.environment.yml       ← conda 配方
    └── leggedGym.requirements.txt      ← pip 补充清单（含手动安装说明）
```

这样换电脑时：`git clone 仓库` → 按 `env/README.md` 操作 → 环境还原完成。

---

# 第五部分：Git —— 记录你每天干了什么

## 5.1 Git 是什么：一个"存档机"

想象你打游戏。**Git 就是存档功能。**

```
写代码  →【存档】→  继续写  →【存档】→  崩了？读档回去
```

每次"存档"（叫 **commit**），git 会做三件事：

1. 把**所有文件的当前状态**拍一张快照
2. 记录**谁存的**（你的名字 + 邮箱）
3. 记录**为什么存**（你写的说明）

**好处：**

- 改坏了可以**随时回到任意一次存档**
- 能看出**每一步改了什么、为什么改**
- 可以**整条历史上传**到 GitHub，多人协作、换电脑续上

### Git ≠ GitHub

| | Git | GitHub |
|---|-----|--------|
| 是什么 | 一个**本地工具**，管理存档 | 一个**网站**，云端的存档柜 |
| 在哪 | 你电脑上 | 网上 |
| 关系 | 负责"存档" | 负责"上传保管" |

**你可以只用 git 不用 GitHub**（纯本地版本管理），但反过来不行。

## 5.2 核心概念对照表

| Git 术语 | 大白话 | 类比 |
|---------|--------|------|
| **repository（仓库）** | 被 git 管理的项目文件夹 | 一个存档文件夹 |
| **commit（提交）** | 存一次档 | 按一次保存 |
| **branch（分支）** | 一条独立的开发线 | 存档的平行世界 |
| **remote（远程）** | 云端那个仓库 | 云盘 |
| **push（推送）** | 本地存档 → 云端 | 上传 |
| **pull（拉取）** | 云端 → 本地存档 | 下载 |
| **clone（克隆）** | 把云端仓库整份复制到本地 | 首次下载 |

## 5.3 三区模型（理解这个，git 就懂一半）

```
① 工作区              ② 暂存区              ③ 本地仓库          ④ 云端
（你正在改的文件）   （本次要打包的清单）  （已存档）        （GitHub）
                  git add          git commit        git push
```

| 区 | 含义 | 操作 |
|----|------|------|
| ① 工作区 | 你实际在编辑的文件 | 正常改文件就行 |
| ② 暂存区 | "这次存档要包含哪些改动" | `git add` |
| ③ 本地仓库 | 已经存好的历史 | `git commit` |
| ④ 远端 | GitHub 上的副本 | `git push` |

**为什么要中间那个"暂存区"？**

因为它让你能**挑着提交**。比如你今天改了 3 个文件，但其中 2 个是关于 A 功能的、1 个是关于 B 的，你可以分两次提交，历史更清晰。

> 日常简单使用：`git add .`（全部）+ `git commit` 就够了。

## 5.4 日常循环（这就是你每天要用的全部）

```bash
cd ~/projects/my_project     # ① 进项目
# ...干活，改文件...
git status                   # ② 看看改了什么（好习惯）
git add .                    # ③ 加入本次存档
git commit -m "说明"          # ④ 存档
git push                     # ⑤ 上传
```

**口诀：改文件 → `status` 看 → `add` 加 → `commit` 存 → `push` 传**

### 建议的提交节奏

不用等到"全部做完"才提交。**每完成一个有意义的小进展就提交一次**，比如：

- 写完一个函数 ✅
- 修好一个 bug ✅
- 跑通一次实验 ✅
- 写完一节笔记 ✅
- 下班前收尾 ✅

**好处**：回头能精确看到"哪一步引入了问题"，出事了能只回退一小步。

## 5.5 怎么写好 commit message（重要）

commit message 就是你的"存档备注"。**半年后回看，全靠它。**

### 基本格式

```
类型: 简短说明（不超过 50 字）
```

### 常用类型前缀

| 前缀 | 含义 | 例子 |
|------|------|------|
| `feat` | 新功能 | `feat: 添加 Go2 机器人的奖励函数` |
| `fix` | 修 bug | `fix: 修正 PPO 优势函数的计算错误` |
| `docs` | 文档 | `docs: 补充训练流程说明` |
| `refactor` | 重构（不改功能） | `refactor: 抽出公共配置到 config.py` |
| `chore` | 杂事 | `chore: 更新 .gitignore` |
| `exp` | 实验记录 | `exp: 尝试增大 batch size，效果不佳` |

### ❌ 差的写法 vs ✅ 好的写法

```
❌ 改了一下
❌ update
❌ 1
❌ 测试

✅ fix: 修正 a1 机器人关节限位写反的问题
✅ feat: 添加地形课程学习(curriculum)配置
✅ docs: 补充 legged_gym 学习笔记第 5 章
✅ exp: 对比 PPO 和 SAC 在 rough 地形上的表现
```

**判断标准：半年后你自己看这条记录，能不能立刻想起"当时干了啥、为什么"？**

## 5.6 查看历史：我这些天都干了啥

```bash
git log                          # 完整历史
git log --oneline                # 一行一条，清爽
git log --oneline --graph        # 带分支图
git log --author="你的名字"       # 只看某人的
git log --since="1 week ago"     # 最近一周
git log --since="2026-10-01" --until="2026-10-15"   # 某段时间
```

**实用的"我最近干了啥"命令：**

```bash
git log --since="7 days ago" --pretty=format:"%ad %s" --date=short
```

输出类似：

```
2026-10-15 docs: 添加环境管理与 Git 工作流指南
2026-10-15 docs: 修正环境配方说明
2026-10-14 feat: 跑通 a1 机器人的训练
```

**这就是你的工作日志**——而且是自动生成的、精确到每次改动的。

### 看某次存档具体改了什么

```bash
git show                    # 最近一次
git show 3eb1390            # 指定某次（用 git log 里的短编号）
git show --stat HEAD        # 只看改了哪些文件
```

## 5.7 撤销与回退（安全网）

### 情况一：改乱了，想放弃这次修改（还没 commit）

```bash
git status                        # 先看看改了啥
git restore 文件名                 # 放弃某个文件的修改
git restore .                     # 放弃所有修改（⚠️ 不可恢复）
```

### 情况二：commit 了，想改说明文字

```bash
git commit --amend -m "新的说明"
```

### 情况三：想把某个文件从暂存区拿出来

```bash
git restore --staged 文件名
```

### 情况四：想看某个文件的历史改动

```bash
git log -p 文件名
```

> ⚠️ **新手注意**：涉及 `reset`、`rebase`、`push --force` 这些命令先别碰，容易把历史搞乱。
> 上面这几条足够日常使用了。

## 5.8 用 Git 记录"每天干了什么"的推荐习惯

这是一个可以长期坚持的做法：

1. **每天开工**：`git pull`（如果多人协作），`git status` 看昨天留下什么
2. **每完成一个进展**：立刻 `git add . && git commit -m "类型: 说明"`
3. **每天收工前**：`git push`，把当天的存档传上去
4. **每周回顾**：`git log --since="7 days ago" --oneline` —— 一眼看清这周做了啥

**这样坚持的价值：**

- 你的"工作日志"是**自动生成的**，不用手写
- 记录的是**真实发生的事情**（代码实际怎么变的），不是回忆
- 出问题了能精确定位到某一次改动
- 面试时可以直接展示"我在这个方向持续做了哪些工作"

---

# 第六部分：GitHub —— 逐步完善你的仓库

## 6.1 仓库的第一个门面：README.md

别人（以及半年后的你）打开仓库**第一眼看到的就是 README**。它应该回答：

- 这个仓库是什么？
- 里面有什么？
- 怎么用？

### 一个实用的 README 模板

```markdown
# 仓库名

一句话说明这个仓库是干什么的。

## 目录

| 文件 / 文件夹 | 说明 |
|--------------|------|
| `xxx.md`     | ... |
| `env/`       | 环境配置记录 |

## 环境

- Python 版本
- 主要依赖

## 使用方法

（怎么跑起来）

## 说明

（任何需要备注的事）
```

> 小技巧：GitHub 上**折叠细节**用 `<details>`：
> ```markdown
> <details>
> <summary>点开看详情</summary>
>
> 这里写详细内容
>
> </details>
> ```

## 6.2 `.gitignore`：让 git 自动忽略不该传的东西

有些文件**不应该进仓库**：

| 类型 | 例子 | 为什么 |
|------|------|--------|
| 训练产物 | `logs/`、`*.pt`、`*.ckpt` | 动辄几个 GB，GitHub 有限制 |
| 缓存 | `__pycache__/`、`*.pyc` | 自动生成的，没意义 |
| 虚拟环境 | `venv/`、`.venv/` | 应该在本地重建，不该上传 |
| 密钥 | `*.pem`、`.env` | ⚠️ **绝对不能传** |
| 编辑器配置 | `.vscode/`、`.idea/` | 个人偏好 |

### 怎么用

在仓库根目录建一个 `.gitignore` 文件：

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
venv/
.venv/

logs/
outputs/
wandb/
*.pt
*.pth
*.ckpt

.env
*.pem
*.key

.vscode/
.idea/
.DS_Store
```

### 如果文件已经被 git 跟踪了

`.gitignore` **只对未跟踪的文件生效**。如果某个文件已经被 `git add` 过，需要先移除跟踪：

```bash
git rm -r --cached logs/     # 从 git 移除，但保留本地文件
git commit -m "chore: 停止跟踪 logs 目录"
```

> `--cached` 是关键：它只从 git 的跟踪列表里移除，**不会删你本地的文件**。

## 6.3 项目边界：不要改别人的代码

这是一个非常重要的概念，尤其是做机器人 / 具身智能时。

### 问题场景

你从 GitHub 克隆了 `legged_gym`，然后为了改一个逻辑，直接改了 `legged_gym/envs/base_task.py`。

**会发生什么：**

1. 你的 `origin` 指向 **ETH 官方仓库**，你没有 push 权限 → 推不上去
2. 就算能推，也会把改动混进官方代码
3. 以后 `git pull` 拉官方更新时 → **冲突**

### 正确的心智模型

```
你的电脑上：
├── isaacgym/       ← NVIDIA 的，别动
├── legged_gym/     ← ETH 的，尽量别动
├── rsl_rl/         ← ETH 的，尽量别动
└── my_project/     ← ✅ 你的地盘，随便改
```

**规则：**

- 你的代码**可以调用**别人的代码
- 但**不要修改**别人的代码
- 需要改行为时，**在自己的代码里继承并覆盖**

### 已经改了怎么办？

把改动**导出成记录**，然后还原官方文件：

```bash
cd 别人的仓库
git diff 你改过的文件 > ~/我的仓库/我的修改记录.patch
git restore 你改过的文件          # 还原成官方版本
```

`.patch` 文件是一份"改动说明书"，随时能重新应用，又不会污染官方代码。

### 更实际的做法

如果确实需要长期改，标准做法是 **fork**（在 GitHub 上把别人的仓库复制一份到你名下），然后克隆你自己的 fork。这样你就有 push 权限了。

## 6.4 仓库的目录组织

随着内容变多，建议这样组织：

```
your-repo/
├── README.md                    ← 门面
├── .gitignore                   ← 忽略规则
├── docs/                        ← 各类文档、笔记
│   ├── 环境管理与Git工作流指南.md
│   └── legged_gym_学习笔记.md
├── env/                         ← 环境配置记录
│   ├── README.md
│   ├── xxx.environment.yml
│   └── xxx.requirements.txt
├── src/ 或 代码文件夹/            ← 代码
└── scripts/                     ← 脚本
```

**原则：**

- 根目录**尽量干净**，一眼能看清仓库是干什么的
- 按**类型**分文件夹（文档归文档、环境归环境、代码归代码）
- 文件名用**能自解释**的名字

## 6.5 完善仓库的渐进路线

不用一次做到完美，可以这样逐步来：

| 阶段 | 做什么 |
|------|--------|
| **第 1 步** | 有个 README，说清楚这是啥 |
| **第 2 步** | 加上 `.gitignore`，别把垃圾传上去 |
| **第 3 步** | 按类型分文件夹，根目录清爽 |
| **第 4 步** | 记录环境配方（`env/`） |
| **第 5 步** | commit message 写得规范清楚 |
| **第 6 步** | 重要的笔记/代码加注释和说明 |
| **第 7 步** | 持续提交，让仓库反映你的真实积累 |

## 6.6 分支（了解即可，暂时不用）

**分支 = 平行世界**。默认分支一般是 `main`。

- 简单项目：**一直在 `main` 上干活就行**，不用管分支
- 要做大改动 / 多人协作时：才需要开新分支

```bash
git branch                    # 看看有哪些分支
git branch 新分支名            # 建一个分支
git switch 新分支名           # 切过去
git switch main              # 切回 main
git merge 新分支名            # 把分支合并回 main
```

> 新手建议：**先不碰分支**。等 `add/commit/push` 这套用熟了再说。

---

# 第七部分：排错速查（FAQ）

## 7.1 环境类

### `ModuleNotFoundError: No module named 'xxx'`

**这是最高频的错误。** 排查顺序：

```bash
which python              # ① 我在哪个环境？
pip list | grep xxx       # ② 这个环境里有 xxx 吗？
```

- 如果 ② 有，但 import 还是失败 → 可能是 `LD_LIBRARY_PATH` 问题（见 7.4）
- 如果 ② 没有 → 你在错误的环境里，或者真的没装

### `ImportError: libpython3.8.so.1.0: cannot open shared object file`

**原因**：包本身装了，但它依赖的 `.so` 动态库系统找不到。

**排查**：

```bash
ls $CONDA_PREFIX/lib/libpython3.8*    # 文件在不在
echo $LD_LIBRARY_PATH                  # 环境变量有没有带上
```

**修复**：在环境的 `activate.d` 里加脚本，让它每次自动生效：

```bash
# 文件：$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### "我明明装了，为什么报找不到？"（路径类）

**如果这个包是 `pip install -e .` 装的**，而你把源代码文件夹**移动了位置** —— 路径就断了。

```bash
cat $CONDA_PREFIX/lib/python3.*/site-packages/*.egg-link   # 看看它指向哪
```

解决办法：把文件夹移回原位置，或者重新 `pip install -e .`。

> **教训：`-e` 安装的包，源代码文件夹不要随便移动。**

### conda 装包特别慢 / 卡住

配置国内镜像。conda 用 `~/.condarc`，pip 用 `~/.config/pip/pip.conf`。

```yaml
# ~/.condarc
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  nvidia: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
remote_read_timeout_secs: 600.0
solver: libmamba
```

```ini
# ~/.config/pip/pip.conf
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
```

### 我不小心往 base 里装包了，怎么办

```bash
conda activate base
pip uninstall 那个包
```

以后记住：**先 `conda activate`，再 `pip install`**。

## 7.2 Git 类

### `git commit` 报错 "Please tell me who you are"

**原因**：没配身份。修复：

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

### `git push` 报错 "Permission denied (publickey)"

**原因**：SSH 密钥没配好。排查：

```bash
ssh -T git@github.com     # 测试连接
```

- 输出 `Hi 用户名! You've successfully authenticated` → 正常
- 输出 `Permission denied` → 公钥没加到 GitHub，或者加错了

### `git push` 被拒绝（non-fast-forward）

**原因**：云端有你本地没有的提交（比如你在网页上改过东西）。

修复：

```bash
git pull          # 先把云端的拉下来
git push          # 再推
```

### 文件太大了传不上去

GitHub 单文件限制 100 MB。**训练产物（`*.pt`、`logs/`）本来就不该传。**

```bash
git rm -r --cached logs/          # 停止跟踪
echo "logs/" >> .gitignore        # 加进忽略规则
git commit -m "chore: 停止跟踪 logs"
```

## 7.3 Linux 类（新手最常用的几条）

```bash
pwd                  # 我现在在哪个目录
ls                   # 这个目录有什么（-la 看详细）
cd 目录名             # 进入目录
cd ..                # 回上一层
cd ~                 # 回主目录

mkdir 目录名          # 建目录（-p 可建多级）
cp 源 目标            # 复制文件
mv 源 目标            # 移动 / 改名
rm 文件              # 删文件（-r 删目录，⚠️ 不可恢复）

cat 文件             # 看文件内容
head -20 文件        # 看前 20 行
tail -20 文件        # 看后 20 行
grep 关键词 文件      # 搜索

which 命令           # 这个命令在哪
```

**实用技巧：**

- `Tab` 键**自动补全**路径 —— 少打很多字，也不容易打错
- 上/下方向键翻历史命令
- `Ctrl + C` 中断当前正在跑的程序
- `Ctrl + L` 清屏

---

# 附录：命令速查表

## 环境管理

```bash
conda env list                              # 列出所有环境
conda create -n 名字 python=3.10            # 新建环境
conda create -n 名字 --clone 已有环境         # 克隆环境 ⭐
conda activate 名字                          # 激活
conda deactivate                            # 退出
conda remove -n 名字 --all                   # 删除环境

which python                                # 当前用哪个 Python ⭐
pip list                                    # 当前环境装了啥
conda list                                  # 同上

conda env export | grep -v "^prefix:" > environment.yml   # 导出配方 ⭐
conda env create -f environment.yml                        # 按配方重建
pip install 包名
pip uninstall 包名
```

## Git 日常

```bash
git status                      # 看当前状态 ⭐
git add .                       # 加入暂存区
git commit -m "说明"             # 存档
git push                        # 上传
git pull                        # 下载

git log --oneline               # 看历史
git log --since="7 days ago" --pretty=format:"%ad %s" --date=short   # 我这周干了啥 ⭐
git show                        # 看最近一次改动详情

git restore 文件                 # 放弃修改（未提交）
git restore --staged 文件        # 从暂存区取出
git commit --amend -m "新说明"   # 修改上次的说明
```

## Git 初始化（每个新仓库一次）

```bash
git init                                             # 初始化
git remote add origin git@github.com:用户名/仓库.git   # 关联远程
git branch -M main                                   # 分支改名 main
git push -u origin main                              # 首次推送
```

## 首次配置（每台电脑一次）

```bash
git config --global user.name "名字"
git config --global user.email "邮箱"
git config --global init.defaultBranch main

ssh-keygen -t ed25519 -C "邮箱"
cat ~/.ssh/id_ed25519.pub        # 复制内容贴到 GitHub
ssh -T git@github.com            # 测试
```

---

## 最后：一张图记住全部

```
┌─────────────────────────────────────────────────────────┐
│  环境管理                                                │
│                                                          │
│   有新项目 → 会不会冲突？                                 │
│       ├─ 不会 → 直接用现有环境                            │
│       └─ 会   → conda create --clone robot_base          │
│                                                          │
│   开工前 → conda activate xxx && which python            │
│   收工时 → conda env export > environment.yml            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Git 日常                                                │
│                                                          │
│   改文件 → git status → git add . → git commit -m "..."  │
│                                          → git push      │
│                                                          │
│   每周回顾 → git log --since="7 days ago" --oneline       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  核心原则                                                │
│                                                          │
│   • 一个项目一个环境，判断依据是"依赖冲突"                  │
│   • base 保持干净，先 activate 再 install                  │
│   • 别人的代码别改，自己的东西放自己仓库                     │
│   • environment.yml 是主力，pip freeze 只适合纯 pip 环境    │
│   • -e 装的包，文件夹不能移                                │
│   • commit message 写给半年后的自己看                      │
└─────────────────────────────────────────────────────────┘
```

---

*本文档基于本机实际情况整理（conda 26.x / git 2.34 / Python 3.8~3.10 / Ubuntu 22.04）。*
