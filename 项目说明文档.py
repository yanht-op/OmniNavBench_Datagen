'''
    OmniNavBench项目说明文档
'''
'''
OmniNavBench 是一个基于 Isaac Sim 的具身导航测评框架：
    它加载场景和任务，让导航模型控制机器人完成任务，记录执行轨迹，再计算评测指标。 主要覆盖4类导航任务：目标导航/跟随人/社会导航/具身问答  等组合任务。
三个核心目录：
    - bench/：组织评测、对接模型、执行动作和计算分数。
    - OmniNav/：封装底层仿真能力和通用接口。
    - OmniNavExt/：实现具体机器人、控制器、传感器和场景运行逻辑。

一、项目提供哪些功能
1. 运行多类型、组合式导航任务
   覆盖 PointNav、VLN、ObjectNav、SocialNav、Human Following 和 EQA。一条任务指令可能要求机器人先导航到某处，再跟随人或回答问题，而不是只到达一个终点。
2. 在不同机器人和场景中评测同一个策略
   主要支持 H1 人形机器人、Aliengo 四足机器人、Carter 轮式机器人。项目使用 GRScenes 和 Matterport3D 场景；模型观测可以包括 RGB、深度及其他配置的传感器数据。
3. 接入外部导航模型
   已提供 Uni-NaVid、MTU3D、PoliFormer、OmniNav 等适配器。
   常见部署方式是模型在独立进程中提供 HTTP 服务，仿真端发送观测并接收动作。（这里需要两个进程，一个进程是模型进程，一个进程是仿真进程。了解这两个指令）
   自定义策略也可以直接实现 Python BasePolicy 接口，HTTP 并不是唯一方式。
4. 把模型输出转换为机器人动作（四类模型的动作空间不同，需要根据模型类型选择不同的动作转换策略，也就是适配器）
   支持两种主要执行方式：
       - step_action：前进、左转、右转、停止等动作。
       - waypoint：模型给出目标路点，执行器驱动机器人移动。
   模型决策与底层物理步进分开，因此一个动作可能持续多个仿真步，并非每个物理步都调用一次大模型。
5. 记录轨迹并离线评分
   运行阶段主要保存  位置轨迹、步数、耗时、停止原因  等；
   正式评分由离线评分器完成，包括成功率、路径效率、导航误差、组合任务完成情况、社会行为和问答表现。
   --mode train 表示在训练集上运行评测，不是训练模型。当前仓库没有完整的导航模型训练流水线。

二、根目录里的文件怎么理解
- [README.md](/home/yan/Project2/OmniNavBench/README.md)：项目总体说明、环境安装、数据结构和评测命令。
- [HowtoTestModel.md](/home/yan/Project2/OmniNavBench/HowtoTestModel.md)：已有策略的启动方式、权重参数和 EQA 配置。
- [runBench.py](/home/yan/Project2/OmniNavBench/runBench.py)：主入口。解析命令行参数、选择数据和策略、构造评测配置、调用运行器并输出结果。
- [configs/](/home/yan/Project2/OmniNavBench/configs)：仿真 YAML。aliengoh1_test.yaml 用于 H1/Aliengo，carter_v1_test.yaml 用于 Carter。
- [envsets/](/home/yan/Project2/OmniNavBench/envsets)：示例任务 JSON。用于理解一个任务如何描述场景、机器人、初始位置和导航目标，不是完整 benchmark 数据集。
- [local_paths.env.example](/home/yan/Project2/OmniNavBench/local_paths.env.example)：数据集、场景资产和 Isaac Lab 路径的配置模板。
- [load_local_paths.sh](/home/yan/Project2/OmniNavBench/load_local_paths.sh)：加载本机路径配置，设置环境变量和 PYTHONPATH。
- [omninav.python.kit](/home/yan/Project2/OmniNavBench/omninav.python.kit)：Isaac Sim 应用配置，声明摄像头、动画、导航网格和 Matterport 等需要加载的扩展。
- [benchtestbatch.sh](/home/yan/Project2/OmniNavBench/benchtestbatch.sh)：按场景、机器人分配批量任务，支持多 GPU。
- [benchteststyle.sh](/home/yan/Project2/OmniNavBench/benchteststyle.sh)：遍历不同语言指令风格。
- [pyproject.toml](/home/yan/Project2/OmniNavBench/pyproject.toml)：Python 项目信息和依赖。
之前生成的 DEPLOYMENT_SERVER_ZH.md 是我们的部署说明，不属于原始评测执行链。

三、bench/：最值得先看的评测层
这是入门的重点目录。
- [evaluator/](/home/yan/Project2/OmniNavBench/bench/evaluator)：评测运行与评分
  bench_runner.py 中的 BenchRunner 负责加载任务、按场景组织执行、管理仿真生命周期和汇总结果。
  episode_runner.py 中的 EpisodeRunner 负责单个 episode 的循环：获取观测、调用策略、执行动作、记录轨迹、判断结束。
  termination.py 管理结束条件；runtime_limits.py 管理任务时间预算；eqa_runtime.py 管理问答能力的显式启用与调用。
  offline_test.py 是理解最终评分口径的关键入口。它读取结果轨迹和带 GT 的标注进行评分。
- [policy/](/home/yan/Project2/OmniNavBench/bench/policy)：模型接入层
  base.py 定义三个核心对象：
        Observation  # 图像、深度、机器人状态、语言指令等输入
        Action       # 速度、离散动作、停止标记等输出
        BasePolicy   # 策略接口，核心方法为 act(observation)





'''




































































