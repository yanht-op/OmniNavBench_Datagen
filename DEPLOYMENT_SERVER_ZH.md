# OmniNavBench 服务器部署方案（单卡 NVIDIA L40）

编写日期：2026-10-14。代码审阅基准：本地提交 `f997cce539505af141c6acef48aed31a1f3314b7`。本文是部署方案，未连接 worker-0，也未在其上执行安装或 GPU 验证。

## 1. 推荐路线与服务器判断

采用 Linux + Conda 原生安装，固定 Python 3.11、Isaac Sim 5.0.0、Isaac Lab 2.3.0。先跑单个 Carter/forward episode，再验证 H1/Aliengo、离线评分，最后安装目标策略并做批量评测。仓库 Docker 发布仍在 TODO 中，因此第一阶段不依赖项目官方镜像。

你提供的 worker-0 有一张 NVIDIA L40；`nvidia-smi` 显示总显存 46068 MiB、已使用 7815 MiB，截图时约剩 37.4 GiB。L40 具备 RT Core，适合本项目的 RTX/Vulkan 仿真要求。现有 PID 41934 的 Python 进程应保留；开始运行前重新确认可用显存，先只启动一个仿真进程。[NVIDIA L40 官方规格](https://www.nvidia.com/en-us/data-center/l40/)

驱动为 570.195.03，显示 CUDA 12.8。这个 CUDA 字段不能证明已安装 CUDA Toolkit，也不能证明 Vulkan 渲染正常。先保留驱动，通过下文 CUDA、Vulkan 和 Isaac Sim 三层验证判断；如出现驱动兼容问题，再由管理员安排升级，不在共享服务器上直接更换驱动。当前 Isaac Lab 文档推荐较新的生产驱动，但不能据此断言现有驱动无法运行固定的 Sim 5.0.0。[Isaac Lab 安装要求](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html)

尚未确认操作系统、CPU、RAM、存储配额、是否处于容器中。建议 Ubuntu 22.04 x86_64，至少 32 GB RAM；64 GB 及以上更适合完整场景。CPU 16 核、NVMe 和至少 500 GB 起步的工作空间是工程规划建议，不是经过本项目测量的最低配置。完整场景集可能需要更多空间，应先统计下载与解压体积，不能把标注数据仓库大小当成总需求。

## 2. 部署前检查

在 worker-0 执行：

```bash
cat /etc/os-release
uname -m
ldd --version
nproc
free -h
df -h "$HOME"
quota -s
nvidia-smi
command -v conda
command -v vulkaninfo
vulkaninfo --summary
```

`quota` 或 `vulkaninfo` 尚未安装时对应命令可能不存在。重点核对：Linux x86_64、pip 安装 Isaac Sim 所需的 glibc >= 2.35、实际内存与磁盘配额、Vulkan 是否识别 NVIDIA L40。headless 只表示不显示窗口，仍然需要 GPU 图形驱动与 Vulkan。若 worker-0 实际是容器，还要核对 NVIDIA GPU 的 graphics 能力、驱动库和 Vulkan ICD 是否映射进容器。[Isaac Lab pip 安装说明](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)

Ubuntu 上缺少基础工具时，由有权限的管理员安装：

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake unzip tmux vulkan-tools libvulkan1
```

服务器需要访问 GitHub、PyPI、NVIDIA Python 包源、PyTorch wheel 源、Hugging Face，以及 `omninav.python.kit` 中的 Omniverse 扩展源。首次启动可能下载扩展并编译缓存。受限网络应提前准备包和扩展缓存。

## 3. 目录与环境隔离

以下以 `$HOME` 为例；数据盘容量更大时将数据目录换成该盘绝对路径。

```text
~/work/OmniNavBench/             基准代码
~/work/IsaacLab/                 Isaac Lab 2.3.0
~/data/OmniNavBench/             annotations、可选 videos
~/data/Assets/                   场景、机器人、行人等资产
~/models/                       模型权重
```

建立 `omninav-sim` 环境运行仿真；模型推理使用单独的 Conda 环境。两者通过 HTTP 通信，避免模型的 transformers、torch 等依赖与 Isaac Sim 环境冲突。基准部署本身不需要先安装任何大模型。

## 4. 安装固定版本的仿真栈

```bash
mkdir -p "$HOME/work" "$HOME/data/OmniNavBench" "$HOME/data/Assets" "$HOME/models"
conda create -n omninav-sim python=3.11 -y
conda activate omninav-sim
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.0.0' \
  --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

cd "$HOME/work"
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install none

git clone https://github.com/Sun-Season/omni.isaac.matterport.git \
  source/omni.isaac.matterport
test -f source/omni.isaac.matterport/config/extension.toml
git -C source/omni.isaac.matterport rev-parse HEAD
```

`none` 表示不额外安装强化学习训练框架。Matterport 扩展通过 Kit 的扩展目录发现，不能只安装 Isaac Lab 而漏掉这个目录。本文未能在线读取该扩展的完整说明，需以克隆后的 README 和扩展清单核对其额外依赖；记录实际可用提交以便复现。

注意：在线安装教程会更新，本次查看的教程已给出 Sim 5.1.0 示例；本项目 README 和自定义 Kit 标明 5.0.0，因此上面主动锁定 5.0.0，不直接复制教程的最新版本号。项目要求 torch 2.7.*、numpy 1.26.*，后续安装后要再次核验版本。

先验证基础仿真：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
cd "$HOME/work/IsaacLab"
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

首次启动按提示阅读并接受 NVIDIA EULA。空场景能持续运行且没有致命 GPU/Vulkan 错误后，Ctrl+C 结束。此验证只证明基础仿真栈可用，后续还要验证项目自定义 Kit 和资产。

## 5. 安装项目及处理已发现的打包问题

```bash
cd "$HOME/work"
git clone https://github.com/AutoLab-SAI-SJTU/OmniNavBench.git
cd OmniNavBench
git rev-parse HEAD
```

本次审阅的 `pyproject.toml` 未显式配置多个顶层包的发现方式。我在临时目录用同样的配置和顶层包结构复现了 setuptools 错误：`Multiple top-level packages discovered in a flat-layout`。这不是在 worker-0 完整安装得到的结论，但足以作为安装前检查项。

如果服务器克隆版本仍缺少相应配置，在 `pyproject.toml` 末尾加入以下内容；如新版已有配置，应合并而非重复追加：

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["bench*", "OmniNav*", "OmniNavExt*", "datagen*"]
```

随后：

```bash
python -m pip install -e .
python -m pip install addict onnxruntime lcm
python -m pip check
python -m pip show isaacsim isaaclab torch torchvision numpy
```

补充依赖的代码依据：`OmniNav/macros.py` 导入 `addict`；G1/GR1 控制器导入 `onnxruntime`；GR1 遥操作控制器导入 `lcm`。它们不在根目录依赖列表内，而 `OmniNavExt/controllers/__init__.py` 会批量导入控制器，因此只运行 Carter 也可能触发这些导入。某些包可能已由 Isaac 栈间接安装，仍应显式核验；它们并不构成经过完整启动验证的全部依赖清单。

`pip check` 非零时先阅读冲突双方的约束，不要忽略，也不要盲目升级整个环境。当前仓库没有完整锁文件；以固定 Isaac 版本和项目约束求得可运行组合后保存 `pip freeze`。

## 6. 准备数据：标注、机器人、行人、场景分别处理

Hugging Face 数据仓库当前根目录包含 `annotations/`、`videos/`、`robots.zip` 和 `People.zip`；页面显示合计约 41.2 GB，其中 People.zip 约 8.55 GB、robots.zip 约 214 MB。该体积不包含另行准备的完整 GRScenes / Matterport3D 场景集。[数据文件列表](https://huggingface.co/datasets/AutoLab-SJTU/OmniNavBench/tree/main)

首轮不下载视频：

```bash
hf download AutoLab-SJTU/OmniNavBench --repo-type dataset \
  --include 'annotations/**' 'robots.zip' 'People.zip' \
  --local-dir "$HOME/data/OmniNavBench"

unzip -l "$HOME/data/OmniNavBench/robots.zip" | head -40
unzip -l "$HOME/data/OmniNavBench/People.zip" | head -40
```

先看压缩包内部目录，再解压到正确层级，避免形成 `Assets/robots/robots/...`。行人资产的目标位置必须与实际 annotation 的 `virtual_humans.asset_root.path/fallback` 一致，不能仅凭压缩包名猜测。需要 `Biped_Setup.usd`、角色 USD、动画和它们引用的相关资源。

最终场景根目录至少应具备所选 episode 引用的资产：

```text
Assets/
├── robots/
│   ├── Carter/carter_v1.usd
│   ├── h1/h1.usd
│   │   └── （同级 policy/move_by_speed/h1_loco_model_20000.pt）
│   └── aliengo/aliengo_camera.usd
│       └── （同级 policy/move_by_speed/aliengo_loco_model_4000.pt）
├── GRScenes/
│   ├── commercial_scenes/{scenes,Materials,models}/
│   └── home_scenes/{scenes,Materials,models}/
├── matterport_usd/<scene_id>/<scene_id>.usd
└── IsaacAssets/Environments/...
```

上面的 policy 文件是机器人运动控制权重，和视觉导航模型权重不同；forward 运行 H1/Aliengo 时仍需要它们。行人目录按 JSON 的实际引用补齐。

场景准备步骤：

1. 从一个 train/original/car 标注文件入手，检查 `scene`、`robots`、`virtual_humans`，只准备该 episode 所需的场景与依赖。
2. GRScenes 从项目链接的 [GRUtopia/GRScenes 上游](https://github.com/OpenRobotLab/GRUtopia) 获取，保留相对材质和模型路径。
3. Matterport3D 按[官方获取流程](https://github.com/niessner/Matterport)准备；原始扫描文件不等于项目需要的 USD。若只有原始 OBJ/纹理，必须完成转换，并验证坐标、尺度和引用与 benchmark annotations 匹配。不能用另一套场景替代后仍声称是同一个 benchmark。
4. 当前仓库说明没有提供一条完整获取“所有 benchmark 对应 USD 场景”的可复现命令。若没有对应场景包或转换流程，需要向维护者确认具体版本与资源；这可能是完整部署的主要阻塞点。

不要把 `OmniNav/download_assets.py` 当作整个 benchmark 的一键下载器：它面向 GRScenes，且使用较旧的下载 CLI/版本；它不会自动替你完成所有 Matterport 场景准备。

## 7. 设置路径与静态检查

创建 `local_paths.env`，填写真实绝对路径，例如：

```bash
OMNINAV_BENCH_DATASET_ROOT="/home/s2510229/data/OmniNavBench"
OMNINAV_SCENE_ROOT="/home/s2510229/data/Assets"
OMNINAV_ISAACLAB_SOURCE="/home/s2510229/work/IsaacLab/source"
```

用户名相同也不代表 home 路径必然相同，先用 `echo "$HOME"` 确认。

```bash
cd "$HOME/work/OmniNavBench"
source load_local_paths.sh
python -c 'from OmniNav.local_paths import resolve_runtime_kit_path; print(resolve_runtime_kit_path())'
python runBench.py --help
```

路径解析会检查 `IsaacLab/source/omni.isaac.matterport` 并在临时目录生成 Kit。`--help` 成功仍不代表完整扩展和场景可启动。

本地工作区的 `load_local_paths.sh` 有未提交的前置三引号内容，位于 shebang 之前；这属于当前工作区修改，不是上游原始脚本。若把此目录同步到服务器，需要先检查并移除错误前缀，使第一行是 `#!/usr/bin/env bash`。本次仅记录问题，没有修改该文件。

## 8. 先跑单 episode 的 forward 冒烟验证

推荐先用 Carter，减少 H1/Aliengo 运动控制权重带来的排障变量。下面 `--envset` 直接接收单个标注文件；具体机器人由文件定义，不需要再用 `--omninavbench` 拼目录。

```bash
conda activate omninav-sim
cd "$HOME/work/OmniNavBench"
source load_local_paths.sh
mkdir -p logs

# 将此路径替换为已经备齐资产的真实文件，不要原样保留 SCENE。
EPISODE="$OMNINAV_BENCH_DATASET_ROOT/annotations/train/original/car/SCENE/final_episode_0.json"
test -f "$EPISODE"

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python runBench.py \
  --envset "$EPISODE" \
  --config configs/carter_v1_test.yaml \
  --scene-root "$OMNINAV_SCENE_ROOT" \
  --output results/smoke_carter_train \
  --policy forward --headless \
  2>&1 | tee logs/smoke_carter_train.log
```

验收条件：

- 自定义 Kit 和 Matterport 扩展加载成功，场景/纹理/机器人正常加载；有行人的场景还需验证动画与 NavMesh。
- episode 真正执行并结束，生成该场景结果 JSON 和 `summary.json`。
- 结果中 `steps > 0`，`trajectory` 非空，位置确实变化，`termination_reason` 合理；timeout 对一直前进的 forward 并不意味着部署失败。
- 需要检查视觉效果时另开一个结果目录并加入 `--record-video`，确认 RGB 画面不是全黑或缺失场景。

不要加入 `--no-trajectory` 或 `--no-save-per-episode`，后续评分依赖逐 episode 轨迹。README 的 `--omninavbench --mode test ...` 快捷命令会读取整个所选目录，不是只跑一个 episode。首次验证应使用上面的单文件方式。

## 9. 评分闭环与其他机器人

```bash
python -m bench.evaluator.offline_test \
  --private "$EPISODE" \
  --results results/smoke_carter_train \
  --output results/smoke_carter_train/scoring.json
```

先确认评分器能匹配该 episode 并输出指标，forward 得分低是预期，不是安装失败。运行阶段的 summary 本来就没有 SR/SPL；评分是独立的离线步骤。

接着分别用 `annotations/train/original/human/...` 和 `dog/...` 的单个标注验证 H1 与 Aliengo，两者都使用 `configs/aliengoh1_test.yaml`。Carter 始终用 `configs/carter_v1_test.yaml`。Linux 路径区分大小写，标注中的 `robots/Carter` 与某些默认配置里的 `robots/carter` 存在差异，要按实际采用的 USD 路径核对。

test 标注不含评分所需私有 GT，不能把 train 的评分方式直接用于 test。test 运行输出应提交官方平台；公开 test annotation 不应修改，否则其哈希与已发布的时间预算可能不匹配。

## 10. 接入模型服务（可选的第二阶段）

确认目标模型后，按模型自身仓库说明建立独立环境，下载匹配权重。OmniNavBench 内的 `OmniNav/` 是仿真集成模块；`--policy omninav` 需要另一个外部 OmniNav 模型仓库与权重，两者不要混淆。

以仓库已有 OmniNav 适配器为例，以下命令假定模型环境已经准备完毕：

```bash
# 终端 A：模型环境，环境名按实际创建的名称替换。
conda activate omninav-policy
CUDA_VISIBLE_DEVICES=0 python "$HOME/work/OmniNavBench/bench/policy/omninav/omninav_server.py" \
  --model_path "$HOME/models/OmniNav/checkpoint" \
  --omninav_path "$HOME/work/OmniNav" \
  --host 127.0.0.1 --port 8005
```

```bash
# 终端 B：仿真环境。
conda activate omninav-sim
cd "$HOME/work/OmniNavBench"
source load_local_paths.sh
curl --fail http://127.0.0.1:8005/health
python runBench.py \
  --envset "$EPISODE" \
  --config configs/carter_v1_test.yaml \
  --output results/omninav_single_train \
  --policy omninav --omninav-server-url http://127.0.0.1:8005 \
  --headless
```

新终端需要重新设置 `EPISODE`。同卡同时容纳仿真、模型和已有 Python 进程时，必须测量峰值显存，不能只根据模型权重大小估算。先单 episode、单仿真进程；模型服务源码使用单线程 Flask，并不能假设增加仿真并发后吞吐线性增长。

EQA 默认关闭。正式评测包含问答任务时，应确认策略有可调用的 `predict_text(question, rgb)` 后再启用 `--enable-eqa`。没有接口时启用会在启动前失败；关闭 EQA 也不会从评分分母中删除问答任务。不能把仅导航成功当作完整任务能力验证。

## 11. 批量运行、日志与复现

单场景、多个 episode 和模型调用均稳定后再运行：

```bash
bash benchtestbatch.sh \
  --robot carter --mode train --style original --policy forward \
  --num-gpus 1 --workers-per-gpu 1 \
  --output-root results/forward_carter_train_batch
```

脚本默认每张 GPU 4 个 worker，对你当前共享使用的单张 L40 不适合直接照搬。保持 1，实测内存与显存后才考虑提升。支持 HTTP 策略时加入对应 `--policy` 和 `--server-url`；模型服务需先启动。

用 `tmux` 保持 SSH 断开后任务继续运行。`logs/batch_*` 下的 `.failed` 日志需要逐一排查；包装脚本会记录单场景失败但继续运行，不能只凭整个 shell 的退出状态判断所有任务成功。`--skip-completed` 主要检查 summary 文件，完整性仍需核对预期 episode 数、实际结果数与轨迹。

不同模型、数据版本、指令风格和配置使用不同结果目录，避免复用旧 JSON。需要强制重跑时，直接运行器支持 `--no-skip`；存在跨 episode PhysX/重置问题时尝试 `--isolate-episodes`，代价是反复启动较慢。

部署成功后保存：

```bash
mkdir -p deployment-records
git rev-parse HEAD > deployment-records/omninavbench.commit
git diff > deployment-records/local-changes.patch
git -C "$HOME/work/IsaacLab" rev-parse HEAD > deployment-records/isaaclab.commit
git -C "$HOME/work/IsaacLab/source/omni.isaac.matterport" rev-parse HEAD \
  > deployment-records/matterport-extension.commit
python -m pip freeze > deployment-records/sim-requirements.txt
conda env export --no-builds > deployment-records/sim-environment.yaml
nvidia-smi > deployment-records/nvidia-smi.txt
```

另外记录数据仓库 revision、场景来源/版本、模型权重版本、完整启动命令和 YAML 配置。不要为正式评测随意改物理步长、图像尺寸、历史长度、超时预算或控制速度，否则会改变可比较性。

## 12. 常见阻塞与完成标准

- `No matching distribution`：首先检查 Python 3.11、x86_64、glibc、包源和指定版本是否可用。
- `Multiple top-level packages`：按第 5 节配置 setuptools 包发现。
- `ModuleNotFoundError`：区分普通 Python 包缺失与 Isaac 扩展尚未启动；不要用随机同名 pip 包替代 `omni.*` 扩展。
- `omni.isaac.matterport` 无法解析：检查 source 路径、扩展清单、版本兼容性与 Kit 加载日志。
- USD、纹理、角色或控制权重缺失：检查 scene-root、压缩包层级、大小写、USD 内部引用和 annotation 路径。
- Vulkan/RTX 创建失败：核对图形驱动、ICD、容器映射；CUDA 可用不等于渲染可用。
- 显存不足：先减少到单仿真进程并关闭录制，协调已有作业，再评估模型独立部署；不要擅自终止 PID 41934。
- HTTP 调用失败：先检查服务日志与 health，再检查模型是否加载、端口与 URL 是否一致。
- 测试集时间预算哈希不匹配：检查代码与公开标注是否同版，不要直接绕过校验。

部署验收依次为：基础 Sim 可启动 → 项目 Kit 和单场景资产加载 → forward 生成有效轨迹 → train 离线评分成功 → 目标模型 HTTP 单 episode 成功 → 批量结果完整。当前 GPU 条件满足开始验证的要求；系统条件及完整场景资产是否已具备，是接下来最需要确认的两项。

## 审阅依据

代码依据为仓库的 `README.md`、`HowtoTestModel.md`、`pyproject.toml`、`runBench.py`、`load_local_paths.sh`、`local_paths.env.example`、`omninav.python.kit`、两套机器人 YAML、`OmniNav/local_paths.py`、`OmniNav/macros.py`、资产下载器、envset 路径与运行时逻辑、控制器注册与权重路径、benchmark runner / offline scorer，以及批量脚本和 OmniNav HTTP 服务。远程项目入口：[OmniNavBench](https://github.com/AutoLab-SAI-SJTU/OmniNavBench)。
