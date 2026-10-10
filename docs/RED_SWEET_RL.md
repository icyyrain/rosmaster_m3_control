# 红色糖果：视觉示教与学习实验（第一阶段）

分支：`codex/red-sweet-rl`。独立工作树：`C:\Projects\rosmaster_m3_control_red_rl`。
此分支从 `experiment/blue-biscuit-grasp` 的 `90c8934` 创建；没有改小车代码，
也没有改变原来的 `pc_client/pick_and_place.py` 控制逻辑。

## 这一阶段做到了什么

`pc_client/red_rl/collect.py` 复用现有规则抓取器作为“老师”，只记录红色糖果的
对齐、下降、最后接近三个阶段。每次命令前后各保存一张相机照片，并记录相机检测的
目标位置/距离/高度、阶段、六关节**最近下发的目标角度**、四关节动作增量及下一次
观测。手腕旋转、夹爪闭合、短距离抬升和低处松开仍由固定程序执行。抬升后操作者
给出成功/失败/不确定标签；相机结果仅作参考，不自动充当可靠奖励。

`train_bc.py` 用**人工确认成功**的完整轨迹做一个线性、带正则化的行为克隆基线，
按完整 episode 分割训练/验证，报告关节 1–4 的角度增量误差。图片保留供后续
视觉编码器使用，当前基线只用相机检测得到的数字特征。失败与不确定轨迹保留在
数据集中，供以后分析和强化学习使用，但不混进模仿目标。

这还不是在线 RL，也**不能**把训练产物直接下发到机械臂。缺少的关键部分是：
安全约束下的学生策略执行、从成功/失败轨迹学习的在线更新、人工随时接管，以及
对红糖果稳健的自动奖励。首轮数据的人工核验不能省略。

示教入口的高位对齐容差为 30 px；原抓取脚本仍为 25 px。实测在糖果偏右的新位置，
整数关节命令于约 26 px 误差处连续量化成零动作而无法满足 25 px。下降阶段每步仍
继续视觉修正。高位对齐失败会回 HOME，不会尝试下降。

## 网络与运行

在 PowerShell 中先切换有线网卡：

```powershell
cd C:\Projects\rosmaster_m3_control
.\car-mode.cmd
ping -S 192.168.2.10 192.168.2.4
```

`ping -S` 强制走小车有线网卡。电脑上运行 FlClash 等虚拟网卡时，普通端口连接可能
被它接走而假报“可连接”。要确认来自小车的真实 SSH banner/rosbridge 握手，
不能仅看 TCP 端口测试。2026-10-10 本次切换后本机有线地址是
`192.168.2.10/24`。首次检查时 `.3` 和 `.4` 没有 ARP 应答；用户恢复小车链路后
`.4` 已连通。若 SSH 可用而 9090 不通，按主 README 启动 rosbridge 和相机驱动。

在独立工作树中运行（使用主工作树已有的 Windows venv）：

```powershell
cd C:\Projects\rosmaster_m3_control_red_rl
& C:\Projects\rosmaster_m3_control\.venv\Scripts\python.exe -m pc_client.red_rl.collect --fast
```

不带 `--execute` 只检查相机、深度和目标，**不发机械臂动作**。把红糖果摆在安全
可达范围且预检通过后，清空周围障碍并准备急停，再运行：

```powershell
& C:\Projects\rosmaster_m3_control\.venv\Scripts\python.exe -m pc_client.red_rl.collect --fast --execute
```

程序要求二次确认，然后回 HOME、用规则控制器接近、闭爪、短抬升、询问是否夹住、
下降至抓取高度约 10 mm 上方松开、回 HOME。每次示教保存到
`datasets/red_sweet/<时间>-<随机码>/`，其中 `episode.json` 有人工标签，
`steps.jsonl` 有完整转移，`images/` 是命令前后图片。数据和模型目录被 Git 忽略；
建议另外备份，不要把实验图片误提交到仓库。

已有离线模型时，可以加 `--shadow-model models/red_sweet_bc.npz`。模型只在每步
计算“如果由我控制会下发什么”，预测值和与规则动作的误差写入同一条记录；真正
发送给机械臂的始终是规则控制器的动作。这个影子模式不会打开连续 MP4 录像；
示教数据仍按步保存前后 RGB 图像，普通预检则不保存这些图片。

收集多个成功 episode 后训练并检查留出轨迹误差：

```powershell
& C:\Projects\rosmaster_m3_control\.venv\Scripts\python.exe -m pc_client.red_rl.validate_dataset
& C:\Projects\rosmaster_m3_control\.venv\Scripts\python.exe -m pc_client.red_rl.train_bc
& C:\Projects\rosmaster_m3_control\.venv\Scripts\python.exe -m unittest tests.test_red_rl_dataset -v
```

模型输出为 `models/red_sweet_bc.npz`，**仅供离线评估**。至少两条成功轨迹才能
得到按 episode 留出的验证结果；一条示教训练出来的低训练误差没有部署意义。

### 2026-10-10 首轮实机结果

在小车 `.4` 的 rosbridge、Orbbec 相机驱动启动后，确认 HOME 视角并完成预检：

| episode | 摆放位置/结果 | 学习阶段步数 | 人工标签 |
| --- | --- | ---: | --- |
| `20261010-101806-d80859` | 原位置；短抬升后夹住 | 28 | 成功 |
| `20261010-102038-2c6cbd` | 糖果偏右；高位对齐卡在约 26 px，没有下降 | 15 | 未判定 |
| `20261010-102306-dc41b4` | 同一偏右位置；30 px 高位容差后短抬升夹住 | 21 | 成功 |

三条数据均通过 `validate_dataset`。用一条成功 episode 训练、另一条整条留出时，
四个关节的单步 MAE 分别约为 `0.70° / 0.73° / 1.25° / 0.69°`。这只证明采集与
离线训练链路正常；两条成功示教远不足以评估泛化，更不能据此直接执行学生策略。
数据和模型都保存在本工作树的 Git 忽略目录中，另行备份后再清理工作树。

## 后续变成强化学习的顺序

1. 收集从 HOME 开始、糖果在安全范围内不同位置的多条成功/失败轨迹；核验时间戳、
   相机图片、人工标签和命令记录。先不要随机初始化机械臂；随机起点必须先定义
   安全关节区域、避碰范围及回 HOME 路径。
2. 用离线行为克隆得到可预测的初始策略，在记录数据上验证；随后加动作增量限制、
   高度下限、视觉丢失停止、超时停止和人工接管，才允许学生策略做少量实机试验。
3. 将 episode 的二值成功奖励和人工接管转成 HIL-SERL / RLPD 风格的
   demonstration + online replay。图像编码器与 actor/critic 可参考
   `C:\Projects\hil-serl-upstream`，但其中的 Franka 环境不能直接复用。
4. 验证自动成功判定之后，再考虑用它替换人工奖励；对相机里被夹爪遮挡或随相机
   视角移动的糖果，要保留“不确定”而非强行判成功/失败。

本车目前没有可信的真实关节反馈。记录中的 `commanded_pose_before/after` 仅表示
最近发布的目标角度，不能当成机械臂实际执行角度；以后接入真实关节反馈时，
需在数据格式中新增独立字段，不能覆盖这个字段。
