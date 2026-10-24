# 多子任务 ObjectNav 训练数据生成

入口 `runDataGen.py` 复用训练 JSON 中的场景、机器人和 NavMesh 配置，清除旧任务与 GT，再采样新起点和对象目标。默认一条 episode 包含 **3 个连续 ObjectNav 子任务**，目标实例及类别不重复：寻找目标 1 → 寻找目标 2 → 寻找目标 3。后一个子任务接续前一个子任务执行，不单独重置机器人。`--num-subtasks N` 调整数量；`--no-distinct-categories` 允许不同实例属于同一类别。无法构建完整任务链或执行失败的样本不会进入训练 manifest。

代码和合成数据测试已实现；尚未在具有完整 GRScenes 资产的 L40 / Isaac Sim 环境完成真实采集验收。测试生成的合成媒体不是实际训练数据。

## 采集过程

1. 从 `OmniNavBenchData_GRScenes/annotations/train/original/<robot>` 读取配置，按场景 USD 路径、单位及根 prim 去重，排除 Matterport。只复用资产配置，不复制旧轨迹作为新 GT。
2. 在 NavMesh 上随机采样起点，为语义对象采样可达的接近位置，构建指定长度的目标链。ObjectNav 的终点由目标对象约束，不是任意空间点。
3. 机器人依次跟踪规划路径，接近目标并转向观察目标；检查到达条件和目标可见像素。导出每个子任务的对象 ID、类别、坐标、指令及视频帧边界。
4. 同步记录机器人执行后的位姿、RGB、深度视频和逐帧 float32 原始深度，按实际时间戳构建未来轨迹训练标签。
5. 校验视频、轨迹、原始深度和任务分段后，发布标准目录并写入 `manifest.jsonl`。

当前指令使用对象语义与模板生成，整体指令按顺序连接各段指令。没有宣称完成复杂语言改写、四种语言风格增强或开放词汇目标消歧。当前支持静态场景、单机器人；需要 USD 中可用的对象语义，或 `--object-annotations` 提供带有效 `prim_path` 的外部对象标注。仅有对象坐标不足以做目标可见性校验。

## 在集群运行

使用项目要求的 Isaac Sim 5.0、Isaac Lab 2.3、Python 3.11 环境和完整 Assets（含机器人控制器依赖）。以下命令在仓库根目录运行，将 `/实际的Assets根目录` 换成 L40 上真实的资产目录。

```bash
python -m datagen.batch \
  --dataset-root /public/home/s2510229/OmniNavBenchData_GRScenes \
  --scene-root /实际的Assets根目录 \
  --config configs/carter_v1_test.yaml \
  --output /public/home/s2510229/OmniNavBenchData_Gen \
  --robot car --episodes-per-scene 1 --max-scenes 1 \
  --num-subtasks 3 --seed 42 --fps 15 \
  --plan-only
```

`--plan-only` 扫描标注、检查命令参数并显示计划，不创建输出目录，也不启动仿真。去掉它执行一个场景的真实采集。首个场景验收成功后，继续使用同一输出目录，设置 `--max-scenes 0` 遍历所有场景，并增加 `--episodes-per-scene`。不自动提交集群作业，继承作业中的 `CUDA_VISIBLE_DEVICES`。每场景独立进程，默认超时 3600 秒，可通过 `--scene-timeout-s` 修改。

`--episodes-per-scene` / 单场景的 `--num-episodes` 是**尝试数量**，不保证成功数量。成功不足时返回退出码 2，但已通过的样本仍可用；失败详情见 `_runs/<运行ID>/` 内的 `batch_summary.json`、场景日志及各场景的 `summary.json`、`attempt.json`。输出目录可以提前创建，也可以包含已有数据；重复运行会追加样本，不覆盖已有 JSON、视频或顶层 manifest。相同 seed 可能重复采样相同任务，追加采集时应更换 seed。

单场景预检查与采集入口：

```bash
python runDataGen.py \
  --config configs/carter_v1_test.yaml \
  --envset /public/home/s2510229/OmniNavBenchData_GRScenes/annotations/train/original/car/MV4AFHQKTKJZ2AABAAAAAEI8_usd/final_episode_3.json \
  --scene-root /实际的Assets根目录 \
  --output /public/home/s2510229/OmniNavBenchData_Gen_smoke \
  --num-episodes 1 --num-subtasks 3 --seed 42 --fps 15 \
  --preflight
```

去掉 `--preflight` 启动采集。预检查检查直接资产引用、配置和 Python 依赖，不验证 GPU、USD 间接引用或目标语义质量。默认 headless，可加 `--no-headless`。Carter 使用 `configs/carter_v1_test.yaml`；H1/Aliengo 使用 `configs/aliengoh1_test.yaml`，批量分别指定 `--robot human` / `--robot dog`，可共用同一输出目录（按机器人目录区分）。

起点采样参数包括 `--min-clearance-m`（默认 0.35）、`--start-min-dist-m`（1.0）、`--object-start-min-dist-m`（2.0）；对象接近位置距离参数为 `--object-goal-min-m`（0.5）和 `--object-goal-max-m`（1.5）。`--no-random-start` 使用原始起点。可以用 `--navmesh-agent-radius-m`、`--navmesh-max-step-height-m` 修正源配置的烘焙尺寸，入口按场景单位换算；参数应匹配真实机器人尺寸。

## 标准输出

```text
OmniNavBenchData_Gen/
  annotations/train/original/car/<GRScenes场景ID>/final_episode_0.json
  videos/train/car/<GRScenes场景ID>/final_episode_0/
    rgb.mp4
    depth.mp4
    depth_raw/
      000000.npy
      000001.npy
      ...
    path.json
    trajectory_targets.jsonl
  manifest.jsonl
  _runs/<运行ID>/
    batch_plan.json             # 批量运行时
    batch_summary.json          # 批量运行时
    <scene_key>.log             # 批量运行时
    manifest.jsonl              # 本次运行的样本索引
    _work/<scene_key>/          # 批量运行时的单场景工作目录
      collection.json
      source_envset.json
      summary.json
      manifest.jsonl
      _episodes/episode_000000/...
```

场景目录沿用输入标注的 GRScenes 场景 ID；首次样本编号从 0 开始，之后按同一机器人/场景目录的已有最大编号递增。编号同时检查 JSON 和视频目录，避开中断运行留下的媒体目录。发布过程使用文件锁协调并发编号和 manifest 追加；已有文件不会被覆盖。

`_runs` 保留各次采集与诊断中间产物（包含媒体副本），不作为额外训练样本。单场景运行的 `collection.json`、`source_envset.json`、`summary.json` 和 `_episodes` 直接位于 `_runs/<运行ID>/`。启动日志会打印本次诊断目录。顶层 `manifest.jsonl` 汇总所有运行发布的样本；运行目录中的 manifest 只包含该次采集结果。

只通过顶层 **`manifest.jsonl`** 读取训练样本，里面的路径相对数据集根目录。JSON 保留场景/机器人配置，并包含：

- `task.navigation.instruction`：整体有序任务指令。
- `task.subtasks`、`task.sub_instructions`：各子任务目标及指令。
- `datagen_segments`：目标实例、规划路径、实际结束位置、起止视频帧及仿真时间。
- `recording.gt_path` 和 `robots.entries[0].rb_gt_waypoints`：相同的执行轨迹，兼容两种读取方式。
- `media`：RGB、深度、原始深度、轨迹和训练标签的相对路径及编码说明。

`gt_path[i]` 对应视频第 `i` 帧和 `depth_raw/{i:06d}.npy`。`frame` 从 0 连续编号；`time_s` 是仿真时间，`sim_step` 是物理步号。任务结束可能补录一帧，因此不要直接以 `frame / fps` 替代真实时间。轨迹坐标转换为米需要乘 `scene.units_in_meters`。

`depth.mp4` 是 8 bit 深度可视化，不适合作为精确深度标签；训练应读取 `depth_raw/*.npy`。后者原样保存相机 `distance_to_image_plane` annotator 的 float32 数组，保留 NaN/Inf/0，读取时采用 `isfinite(depth) & (depth > 0)` 有效掩码。原始深度的单位沿用传感器输出，正式训练前应在实际场景核验标定与单位。

## 轨迹模仿学习

可使用“当前/历史 RGB、原始深度、整体任务指令 → 未来局部轨迹”训练策略。`trajectory_targets.jsonl` 每行关联观测帧，包含：

- `future_xy_local_m`：未来 0.5、1.0、1.5、2.0 秒的位置，以当前机器人为原点，x 向前、y 向左，单位米；根据实际时间戳插值。
- `valid`：超出整条 episode 的 horizon 为 false，训练损失必须掩码；所有 horizon 无效的行应跳过。
- `subtask_index`、`subtask_end`：当前帧所属子任务及该段最后一帧。
- `is_last_observation`：整条 episode 的最后观测，不代表真实控制器 stop 指令。

未来轨迹允许跨子任务边界，适用于以整体指令为条件的连续任务训练；若只输入当前子指令，需要额外屏蔽跨段的 horizon。`gt_path[].command` 是位姿差分反推的速度估计，不能当作真实下发的底层控制动作。规划参考路径与执行后的 GT 分别保存，不能混用。自动控制轨迹与官方人工遥操作轨迹的分布可能不同。

样本校验逐帧解码两类视频，核对帧数、轨迹时间、两份 GT、原始深度数量/尺寸/dtype、子任务数量、目标唯一性及分段连续性。只对完整、通过校验的任务链生成训练 manifest。训练/验证应按场景划分；不要用官方 test 场景扩充训练集。

单场景还保留辅助 `--task pointnav` 模式用于自由可达起终点采样，使用几何路线模板指令；批量 GRScenes 默认固定为多子任务 ObjectNav。

## 不依赖仿真的测试

初始化长时间无进展时，查看当前 `_runs/<运行ID>/progress.log` 的最后阶段，以及 `python_stacks.log` 中每 120 秒自动记录的 Python 线程调用栈。此诊断在进程内部执行，不需要 sudo 或附加进程权限。栈文件中的 `Timeout (0:02:00)` 只是周期性采样标题，不表示采集被超时终止；它记录 Python 调用位置，不能直接显示原生 GPU/PhysX 内部调用栈。

在装有 NumPy、OpenCV、PyYAML 的环境中运行：

```bash
python -m unittest discover -s tests -p 'test_datagen*.py' -v
```

测试覆盖真实文件编码/解码、三目标链采样、分段标签、原始深度保存、标准目录发布、不完整样本拒绝及防覆盖。真实 Isaac Sim 运行仍需在集群验收。
