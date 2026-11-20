# 实验日志

## 11/15
### 搭建项目环境：
1. 问题：环境版本不匹配

2. 解决办法：换一个被广泛支持的版本。问了deepseek，现在 Python 3.14.4这个版本太新，做 AI 的那些库（PyTorch、numpy 等）还没来得及适配它，一装就是红色报错。

3. 做了什么：建了 GitHub 仓库，装了 Python 3.11

4. 发现什么：nvidia-smi 命令不存在，电脑没有独立显卡，wmic 命令不存在（AI解释Win11 已移除）。

5. 疑惑：走 CPU 路线，把"零 GPU"作为低成本设计的核心卖点。没有显卡 -> 低成本？

### 粗略了解视频检索流程：
① 数据准备        ② 特征提取          ③ 建索引        ④ 查询          ⑤ 返回结果
视频文件   →   每一帧变成一串数字   →   存进数据库   →   你打字搜索   →   最像的片段
（解码/抽帧）      （通常要GPU）       （向量索引）     （同样变数字）   （按相似度排序）

1. 学习到什么：数据准备过程中解码很费 CPU，存原始帧很占硬盘存原始帧占硬盘；特征提取是全流程所需成本最大的（推测，待实验验证：测每帧耗时 × 帧数），主要在于要对每一帧都要跑一次模型推理；建索引也很占内存，同时索引本身也有大小；查询和返回结果在前面流程搭好的前提下成本约等于0

2. 疑惑：
CLIP？量化在图像视频中是什么？PQ量化？

3. 解决办法：
抽帧：30 帧/秒 -> 只抽 1 帧/秒，计算量减少30倍，从而让存储减少30倍；存在问题：快速动作可能漏掉
特征提取：用现成的模型，训练成本=0；存在问题：精度不如专门训练的

4. 进一步了解：
（1）电脑不认识文字意思，理解不了所表述的内容。图片对于电脑来说只是一些像素数字。所以把图片和文字都翻译成同一种“数字坐标”，让意思相近的东西，坐标也相近。这样就可以完成图文匹配。

（2）数字坐标不是像平面坐标那样简单的两个数，而是 512 个数，即“512 维向量”。它也是索引中的向量。（为什么要512个数字，不能小点吗？也是“成本-精度”的权衡？）

（3）用OpenAI训练的CLIP，实现“零样本”（zero-shot）：模型从没见过你的视频，也没见过你的查询词，但是由于已经经过了大量训练，可以直接使用。

（4）向量占内存，一帧 = 512 × 4字节 = 2048 字节 ≈ 2 KB。检索时要把这些数字全读进内存进行反复比对，成本大，效率低。

（5）量化的本质是用“编号”代替“精确值”，换空间。（但是这样的话丢失了细节，如果多个事物同类似于一个编号，容易分不清它们）

（6）PQ量化：“8 个 64 维” 拼起来的表达能力>>“1 个 512 维” 用同样多的代表
我不需要准备所有组合的样本，只需要准备每个分段的样本。零件库小，成品空间大。空间压缩又保留精度。一个字节正好256个值，所以每个分段256个代表。如果是512个，那就要用2个字节存一个编号，不划算。
这一方法会使召回率（有没有漏掉）下降其中所分的段数可调，需要平衡成本与精度
PQ段数越多越准（推测，待验证）


## 11/16
### 文献阅读：
1. 论文文件：
- notes/papers/craw.pdf（PVLDB 2026）
- notes/papers/lovo.pdf（ICDE 2025，arXiv 2507.14301）
- notes/papers/cep.pdf（SIGMOD 2025）

2. 文献笔记：lit.md

### 数据源与代码落地
- 硬件确认：i5-1155G7 4C/8T + 15.8GB，无独显。
- 网络实测：hf-mirror.com OK / modelscope OK / aliyun pypi OK / huggingface.co 不通 / raw.githubusercontent.com 不通。
- 数据源定为 sayakpaul/ucf101-subset（UCF-101 官方子集，171MB，class 文件夹名即标签），走 hf-mirror 下载。
- 代码落地：prepare_data.py / build_index.py / search.py / evaluate.py / README.md。检索用 numpy 暴力点积（万帧量级足够），FAISS 留作后续索引消融。

### 跑通小数据
- cmd 和 PowerShell 是两个不同的壳，命令不通用。
## 2026-11-16 pip 安装踩坑（写进报告 Challenges）
- 现象：pip install -r code/requirements.txt 报 UnicodeDecodeError: gbk codec can not decode byte 0xac。
- 原因：requirements.txt 里有中文注释（UTF-8），而 pip 在中文 Windows 下用 GBK 读该文件。
- 修复：把 requirements.txt 改成纯 ASCII（校验 268 字节、0 个非 ASCII 字节）。
- 教训：被外部工具读取的配置文件一律纯 ASCII；Python 源码本身不受影响（默认 UTF-8）。

## 2026-11-16 断网与断点续传
- 现象：预取权重报 getaddrinfo failed；prepare_data 报 ContentTooShortError（73.4/171.4 MB）。
- 原因：客户端断网，非代码缺陷。
- 修复：新增 code/dl.py（HTTP Range 断点续传 + 自动重试，最多 300 次），prepare_data.py 与新增 get_model.py 全部改用。
- 另一发现：open_clip 走 huggingface_hub 只重试 5 次即放弃，故改为先落地本地权重文件再加载。
- 可写进报告：弱网环境下下载的可复现性设计（resumable download）。

## 2026-11-17 首次跑通（UCF-101 子集）
- 数据：405 段 / 10 类 / 每类约 40 段（train+val+test 合并）。
- 建库：2462 帧，197.6 秒，80 ms/帧，索引 5.04 MB（2 KB/帧 = 512 维 float32）。
- 查询延迟：文本编码 60~97 ms，向量检索 1.2~5.8 ms。
- 结果：R@1=R@5=R@10=90.0%；随机基线 10.0%/41.1%/65.5%。
- 发现 1（天花板效应）：R@1 已达 90%，增大 K 不再提升，语料无区分力。
- 发现 2：max 与 mean 聚合结果完全相同，设计选择在该语料上不可测。
- 发现 3（失败案例）：查询 basketball dunk 的 top-8 全是 Basketball，分数集中 0.357~0.367，
  无一条命中 BasketballDunk，说明帧级 CLIP 缺少运动信息，无法区分细粒度动作。
- 结论：UCF 子集只能作 sanity check，主实验必须换 MSR-VTT（随机 R@10 仅约 1%，有区分力）。

### UCF-101 子集（主测）| ViT-B-32 (laion2b) | 2026-11-17
#### UCF-101 | ViT-B-32 (laion2b) | 1.0 fps | max | 2026-11-17
建库完成：405 段视频 / 2462 帧
向量矩阵：(2462, 512)，占用 5.04 MB
总耗时：251.1 秒  →  102 ms/帧

[索引] 2462 帧 / 405 段视频 （模型 ViT-B-32，抽帧 1.0 fps）
查询：a person playing basketball
（文本编码 59 ms，检索 2.6 ms）

排名  分数      类别                      命中时间      视频
------------------------------------------------------------------------------
1   0.3229  Basketball              0.0s      v_Basketball_g06_c02.avi
2   0.3020  Basketball              0.0s      v_Basketball_g23_c06.avi
3   0.2977  Basketball              0.0s      v_Basketball_g03_c06.avi
4   0.2976  Basketball              1.0s      v_Basketball_g03_c04.avi
5   0.2973  Basketball              1.0s      v_Basketball_g23_c04.avi

[协议] UCF-101 子集（按类别，每查询多个正确答案）
[索引] 2462 帧 / 405 段视频
[配置] 模型 ViT-B-32，抽帧 1.0 fps，聚合 max
[查询] 10 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       90.0%       10.0%       9.0x
Recall@5       90.0%       41.1%       2.2x
Recall@10      90.0%       65.5%       1.4x

每个类别的首个命中位置（None = 前 K 名里一个都没中）：
  apply eye makeup         首个命中 1     
  apply lipstick           首个命中 1     
  archery                  首个命中 1     
  baby crawling            首个命中 1     
  balance beam             首个命中 1     
  band marching            首个命中 1     
  baseball pitch           首个命中 1     
  basketball               首个命中 1     
  basketball dunk          首个命中 None     <- 未命中
  bench press              首个命中 1  

#### UCF-101 | ViT-B-32 (laion2b) | 1.0 fps | max | 2026-11-17（重复测量 #2）
建库完成：405 段视频 / 2462 帧
向量矩阵：(2462, 512)，占用 5.04 MB
总耗时：207.1 秒  →  84 ms/帧

[协议] UCF-101 子集（按类别，每查询多个正确答案）
[索引] 2462 帧 / 405 段视频
[配置] 模型 ViT-B-32，抽帧 1.0 fps，聚合 max
[查询] 10 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       90.0%       10.0%       9.0x
Recall@5       90.0%       41.1%       2.2x
Recall@10      90.0%       65.5%       1.4x

每个类别的首个命中位置（None = 前 K 名里一个都没中）：
  apply eye makeup         首个命中 1     
  apply lipstick           首个命中 1     
  archery                  首个命中 1     
  baby crawling            首个命中 1     
  balance beam             首个命中 1     
  band marching            首个命中 1     
  baseball pitch           首个命中 1     
  basketball               首个命中 1     
  basketball dunk          首个命中 None     <- 未命中
  bench press              首个命中 1  

### 消融实验（UCF-101 抽帧率）| ViT-B-32 (laion2b) | max | 2026-11-17
#### UCF-101 | ViT-B-32 (laion2b) | 0.5 fps | max | 2026-11-17
建库完成：405 段视频 / 1342 帧
向量矩阵：(1342, 512)，占用 2.75 MB
总耗时：120.5 秒  →  90 ms/帧

[协议] UCF-101 子集（按类别，每查询多个正确答案）
[索引] 1342 帧 / 405 段视频
[配置] 模型 ViT-B-32，抽帧 0.5 fps，聚合 max
[查询] 10 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       90.0%       10.0%       9.0x
Recall@5       90.0%       41.1%       2.2x
Recall@10      90.0%       65.5%       1.4x

每个类别的首个命中位置（None = 前 K 名里一个都没中）：
  apply eye makeup         首个命中 1     
  apply lipstick           首个命中 1     
  archery                  首个命中 1     
  baby crawling            首个命中 1     
  balance beam             首个命中 1     
  band marching            首个命中 1     
  baseball pitch           首个命中 1     
  basketball               首个命中 1     
  basketball dunk          首个命中 None     <- 未命中
  bench press              首个命中 1 

#### UCF-101 | ViT-B-32 (laion2b) | 0.25 fps | max | 2026-11-17
建库完成：405 段视频 / 755 帧
向量矩阵：(755, 512)，占用 1.55 MB
总耗时：62.7 秒  →  83 ms/帧

[协议] UCF-101 子集（按类别，每查询多个正确答案）
[索引] 755 帧 / 405 段视频
[配置] 模型 ViT-B-32，抽帧 0.25 fps，聚合 max
[查询] 10 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       90.0%       10.0%       9.0x
Recall@5       90.0%       41.1%       2.2x
Recall@10     100.0%       65.5%       1.5x

每个类别的首个命中位置（None = 前 K 名里一个都没中）：
  apply eye makeup         首个命中 1     
  apply lipstick           首个命中 1     
  archery                  首个命中 1     
  baby crawling            首个命中 1     
  balance beam             首个命中 1     
  band marching            首个命中 1     
  baseball pitch           首个命中 1     
  basketball               首个命中 1     
  basketball dunk          首个命中 8     
  bench press              首个命中 1 

### MSR-VTT test 1k（主实验）| ViT-B-32 (laion2b) | 1.0 fps | max | 2026-11-17
建库完成：1000 段视频 / 16036 帧
向量矩阵：(16036, 512)，占用 32.84 MB
总耗时：1035.8 秒  →  65 ms/帧

[协议] MSR-VTT test 1k（每查询 1 个正确答案）
[索引] 16036 帧 / 1000 段视频
[配置] 模型 ViT-B-32，抽帧 1.0 fps，聚合 max
[查询] 1000 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       32.2%        0.1%     322.0x
Recall@5       55.6%        0.5%     111.2x
Recall@10      66.5%        1.0%      66.5x

[诊断] 1000 条查询里，正确答案排第 1 的有 322 条

命中示例：
  第1名  a woman creating a fondant baby and flower
  第1名  baseball player hits ball
  第1名  little pet shop cat getting a bath and washed with little brush
  第1名  a naked child runs through a field
  第1名  a man is giving a review on a vehicle

失败示例（正确答案落在前 K 名之外）：
  未命中    the man in the video is showing a brief viewing of how the m  (top1=video9517.mp4)
  未命中    cartoon show for kids  (top1=video7701.mp4)
  未命中    there is someone playing a game in a computer  (top1=video7593.mp4)
  未命中    a guy wearing a black shirt talks and shows a chart on the t  (top1=video8682.mp4)
  未命中    it is the animation cartoon  (top1=video7741.mp4)

### 消融实验（MSR-VTT 抽帧率与聚合）| ViT-B-32 (laion2b) | 2026-11-17
#### MSR-VTT test 1k | ViT-B-32 (laion2b) | 1.0 fps | mean | 2026-11-17
[协议] MSR-VTT test 1k（每查询 1 个正确答案）
[索引] 16036 帧 / 1000 段视频
[配置] 模型 ViT-B-32，抽帧 1.0 fps，聚合 mean
[查询] 1000 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       31.5%        0.1%     315.0x
Recall@5       55.0%        0.5%     110.0x
Recall@10      66.9%        1.0%      66.9x

[诊断] 1000 条查询里，正确答案排第 1 的有 315 条

命中示例：
  第2名  a woman creating a fondant baby and flower
  第2名  baseball player hits ball
  第1名  little pet shop cat getting a bath and washed with little brush
  第1名  goldfish chase each other around a blue tank to music
  第3名  while other friends too try and hitting the basket another is eage

失败示例（正确答案落在前 K 名之外）：
  未命中    a naked child runs through a field  (top1=video8655.mp4)
  未命中    a man is giving a review on a vehicle  (top1=video9677.mp4)
  未命中    a man is singing and standing in the road  (top1=video8251.mp4)
  未命中    man in black shirt is holding a baby upside down and talking  (top1=video9320.mp4)
  未命中    a man extinguishes a fire outside  (top1=video9036.mp4)

#### MSR-VTT test 1k | ViT-B-32 (laion2b) | 0.5 fps | max | 2026-11-17
建库完成：1000 段视频 / 8285 帧
向量矩阵：(8285, 512)，占用 16.97 MB
总耗时：544.1 秒  →  66 ms/帧

[协议] MSR-VTT test 1k（每查询 1 个正确答案）
[索引] 8285 帧 / 1000 段视频
[配置] 模型 ViT-B-32，抽帧 0.5 fps，聚合 max
[查询] 1000 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       31.0%        0.1%     310.0x
Recall@5       56.4%        0.5%     112.8x
Recall@10      65.3%        1.0%      65.3x

[诊断] 1000 条查询里，正确答案排第 1 的有 310 条

命中示例：
  第1名  a woman creating a fondant baby and flower
  第1名  baseball player hits ball
  第1名  little pet shop cat getting a bath and washed with little brush
  第1名  a naked child runs through a field
  第1名  a man is giving a review on a vehicle

失败示例（正确答案落在前 K 名之外）：
  未命中    the man in the video is showing a brief viewing of how the m  (top1=video9517.mp4)
  未命中    cartoon show for kids  (top1=video7701.mp4)
  未命中    there is someone playing a game in a computer  (top1=video7593.mp4)
  未命中    a guy wearing a black shirt talks and shows a chart on the t  (top1=video8682.mp4)
  未命中    there is a man shooting other people in a corridor  (top1=video8815.mp4)

#### MSR-VTT test 1k | ViT-B-32 (laion2b) | 0.25 fps | max | 2026-11-17
建库完成：1000 段视频 / 4344 帧
向量矩阵：(4344, 512)，占用 8.90 MB
总耗时：277.5 秒  →  64 ms/帧

[协议] MSR-VTT test 1k（每查询 1 个正确答案）
[索引] 4344 帧 / 1000 段视频
[配置] 模型 ViT-B-32，抽帧 0.25 fps，聚合 max
[查询] 1000 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       30.7%        0.1%     307.0x
Recall@5       55.0%        0.5%     110.0x
Recall@10      64.1%        1.0%      64.1x

[诊断] 1000 条查询里，正确答案排第 1 的有 307 条

命中示例：
  第1名  a woman creating a fondant baby and flower
  第2名  baseball player hits ball
  第1名  little pet shop cat getting a bath and washed with little brush
  第1名  a naked child runs through a field
  第1名  a man is giving a review on a vehicle

失败示例（正确答案落在前 K 名之外）：
  未命中    the man in the video is showing a brief viewing of how the m  (top1=video9517.mp4)
  未命中    a man is singing and standing in the road  (top1=video7765.mp4)
  未命中    there is someone playing a game in a computer  (top1=video7593.mp4)
  未命中    a guy wearing a black shirt talks and shows a chart on the t  (top1=video8682.mp4)
  未命中    there is a man shooting other people in a corridor  (top1=video8815.mp4)


#### MSR-VTT test 1k | ViT-B-32 (laion2b) | 2.0 fps | max | 2026-11-17
建库完成：1000 段视频 / 22412 帧
向量矩阵：(22412, 512)，占用 45.90 MB
总耗时：1759.8 秒  →  79 ms/帧

[协议] MSR-VTT test 1k（每查询 1 个正确答案）
[索引] 22412 帧 / 1000 段视频
[配置] 模型 ViT-B-32，抽帧 2.0 fps，聚合 max
[查询] 1000 条

指标          本系统         随机基线        提升
------------------------------------------------
Recall@1       32.2%        0.1%     322.0x
Recall@5       56.9%        0.5%     113.8x
Recall@10      66.8%        1.0%      66.8x

[诊断] 1000 条查询里，正确答案排第 1 的有 322 条

命中示例：
  第1名  a woman creating a fondant baby and flower
  第1名  baseball player hits ball
  第1名  little pet shop cat getting a bath and washed with little brush
  第1名  a naked child runs through a field
  第3名  a man is giving a review on a vehicle

失败示例（正确答案落在前 K 名之外）：
  未命中    the man in the video is showing a brief viewing of how the m  (top1=video9517.mp4)
  未命中    there is someone playing a game in a computer  (top1=video7593.mp4)
  未命中    a guy wearing a black shirt talks and shows a chart on the t  (top1=video7412.mp4)
  未命中    there is a man shooting other people in a corridor  (top1=video8815.mp4)
  未命中    it is the animation cartoon  (top1=video9403.mp4)

### 2026-11-17 计时方差（重要方法学记录）
同一配置（UCF-101 | ViT-B-32 | 1.0 fps | max | 2462 帧）三次独立测量：
- 197.6 s (80 ms/帧)  助手运行，紧接着 30 段冒烟测试之后（页缓存部分热）
- 251.1 s (102 ms/帧) 用户第一次运行（冷缓存，后台有下载与编辑器占用）
- 207.1 s (84 ms/帧)  用户重跑（缓存已热，机器较空闲）
中位数 207.1 s，极差 197.6~251.1 = 27%。
结论：建库计时受页缓存与系统负载影响，差异小于约 30% 不构成有效差异；
      报告采用中位数，并在方法一节声明该不确定度。
对结论的影响：抽帧率 1.0 -> 0.25 fps 带来 3.2 倍成本下降，远大于 27% 的噪声底，
      因此"降低抽帧率显著降低成本"这一结论不受计时噪声影响。
详见 results/timing_repeats.csv


### 微基准：每帧成本的阶段分解（2026-11-17）

目的：把"模型约占建库成本 X%"从推断变成实测。
脚本：`code/bench_decompose.py`（解码/模型分开计时，4 遍：解码首次、解码二次、仅模型、端到端）
输出：`results/timing_components.csv`

| 数据集 | 帧数 | 解码(二次) | 预处理+前向 | 端到端 | 模型占端到端 | 两部分占端到端 |
|---|---|---|---|---|---|---|
| MSR-VTT (20 段) | 317 | 5.6 ms | 40.4 ms | 55.0 ms | 73.5% | 83.6% |
| UCF-101 (20 段) | 142 | 6.7 ms | 43.5 ms | 65.2 ms | 66.7% | 77.0% |

结论：
- 编码器调用是解码的 **6~7 倍**，解码只占约 1/10。
- 两次解码 5.3 vs 5.6 ms → 页缓存已是热的，**冷 I/O 未能测到**（如实记录）。
- 微基准的端到端（55 / 65 ms）比全量建库（65 / 84 ms）快 15~22%，落在 27% 噪声底内。
- 两部分之和只解释端到端 77~84%，**余下的未归因**（每段视频开/关开销等）。
- 因此报告不再写"至少 78% 是计算"，改为**分量实测 + 6~7 倍比值**。


### 2026-11-18 创新点 A：按查询类型分层（scene / object / action）
目的：全局 Recall@1 = 32.2% 是平均值，可能掩盖"抽帧率对不同类型问题的伤害不同"。
做法：把 MSR-VTT test 1k 的 1000 条查询按语义分成三类，再对每个抽帧率分别统计 Recall@K。

分类规则（`code/analyze_query_types.py`，关键词决策表，完全可复现，非人工逐条标注）：
- 优先级 action > scene > object
- action = 含物理动作动词（play/run/dance/cook/sing/create/hold/...）；
  若同时出现 talk/speak/explain 等"话语类动词"，则不判为 action（画面通常静止）
- scene = 不含物理动作，但含场景/地点词（beach/stage/kitchen/street/...）
- object = 其余（人物、物品、属性），占比最大
分布：action 375 条（37.5%）、scene 64 条（6.4%）、object 561 条（56.1%）
分类结果与每条查询在四种抽帧率下的名次已存 `results/query_ranks.json`（审计用）。

结果（R@1 / R@5 / R@10，max 聚合，与主实验同一批索引）：

| 类型 | n | 0.25 fps | 0.5 fps | 1.0 fps | 2.0 fps |
|---|---|---|---|---|---|
| action | 375 | 35.5 / 61.1 / 68.8 | 35.2 / 60.8 / 69.3 | 37.3 / 61.1 / 71.5 | 36.5 / 60.3 / 72.3 |
| scene  |  64 | 35.9 / 59.4 / 68.8 | 35.9 / 64.1 / 75.0 | 32.8 / 67.2 / 75.0 | 37.5 / 65.6 / 76.6 |
| object | 561 | 26.9 / 50.4 / 60.4 | 27.6 / 52.6 / 61.5 | 28.7 / 50.6 / 62.2 | 28.7 / 53.7 / 62.0 |
| ALL    |1000 | 30.7 / 55.0 / 64.1 | 31.0 / 56.4 / 65.3 | 32.2 / 55.6 / 66.5 | 32.2 / 56.9 / 66.8 |

（ALL 行与 results.csv 主结果逐位一致 —— 这是分层的自检：分组不改变总量。）

配对检验（同一批查询，R@1，1.0 fps 分别对比 0.25 / 0.5 / 2.0 fps，McNemar 1 自由度）：
- action：b/c = 23/16 (p=0.34)、20/12 (p=0.22)、14/11 (p=0.69)
- scene ：5/7 (p=0.77)、1/3 (p=0.62)、1/4 (p=0.37)
- object：29/19 (p=0.19)、16/10 (p=0.33)、16/16 (p=0.86)
- 二分对比（action vs 其余）：action 23/16 (p=0.34)、20/12 (p=0.22)、14/11 (p=0.69)；
  其余 34/26 (p=0.37)、17/13 (p=0.58)、17/20 (p=0.74)
结论：**没有任何一类的抽帧率效应显著**（全部 p > 0.19）。方向上 action 类在 0.25 fps 掉得最多
      （37.3 → 35.5，-1.8 pp），但该类的标准误是 2.5 pp，属于噪声范围。

顺带发现（比"抽帧率"更有信息量）：**同一抽帧率下，类型差异大于抽帧率带来的差异**。
1.0 fps 时 action 37.3% vs object 28.7%，相差 8.6 pp（各自 s.e. 2.5 / 1.9，仍属边缘）；
而任何一类在 0.25~2.0 fps 之间的波动都 < 2 pp。
→ 报告的写法应是"召回率由**问什么**决定，而几乎不由**抽多少帧**决定"。

纪律：
- 数字同时写入 `results/results.csv`（dataset = msrvtt_1k_qtype_{action,scene,object}，4×3 = 12 行）
  与本节；两处一致才算数。
- scene 只有 64 条、s.e. 约 6 pp，正文里只作描述性提及，不做显著性主张。
- 分类是规则法而非人工标注，正文需如实说明，并作为限制（label noise 未知）。

[查询] 1000 条，规则分类结果：
  action   375 条  (37.5%)
  scene     64 条  (6.4%)
  object   561 条  (56.1%)

[样例] 每类前 5 条（人工扫一眼就能判断规则是否合理）：
  --- action ---
    a woman creating a fondant baby and flower
    baseball player hits ball
    little pet shop cat getting a bath and washed with little brush
    a naked child runs through a field
    a man is singing and standing in the road
  --- scene ---
    man in black shirt is holding a baby upside down and talking about uni
    a man is talking about making it easier for kids to learn while scenes
    some people video conferencing as they watch a movie
    tennis players are involved in matches in large stadiums in front of l
    a man talks about the school system
  --- object ---
    a man is giving a review on a vehicle
    the man in the video is showing a brief viewing of how the movie is st
    cartoon show for kids
    goldfish chase each other around a blue tank to music
    a guy wearing a black shirt talks and shows a chart on the tv screen

[编码] 查询向量 (1000, 512)

====================================================================================
  fps type       n     R@1     R@5    R@10  s.e.(R@1)
------------------------------------------------------------------------------------
 0.25 action   375  35.5%  61.1%  68.8%       2.5
 0.25 scene     64  35.9%  59.4%  68.8%       6.0
 0.25 object   561  26.9%  50.4%  60.4%       1.9
      ALL     1000  30.7%  55.0%  64.1%
------------------------------------------------------------------------------------
  0.5 action   375  35.2%  60.8%  69.3%       2.5
  0.5 scene     64  35.9%  64.1%  75.0%       6.0
  0.5 object   561  27.6%  52.6%  61.5%       1.9
      ALL     1000  31.0%  56.4%  65.3%
------------------------------------------------------------------------------------
    1 action   375  37.3%  61.1%  71.5%       2.5
    1 scene     64  32.8%  67.2%  75.0%       5.9
    1 object   561  28.7%  50.6%  62.2%       1.9
      ALL     1000  32.2%  55.6%  66.5%
------------------------------------------------------------------------------------
    2 action   375  36.5%  60.3%  72.3%       2.5
    2 scene     64  37.5%  65.6%  76.6%       6.1
    2 object   561  28.7%  53.7%  62.0%       1.9
      ALL     1000  32.2%  56.9%  66.8%
------------------------------------------------------------------------------------

[配对检验] 同一批查询、R@1，1 fps 对比其他抽帧率（McNemar，1 自由度）
类型      对比                b    c        p
action  1.0 vs 0.25 fps   23   16    0.337
action  1.0 vs 0.5 fps   20   12    0.216
action  1.0 vs 2 fps     14   11    0.689
scene   1.0 vs 0.25 fps    5    7    0.773
scene   1.0 vs 0.5 fps    1    3    0.617
scene   1.0 vs 2 fps      1    4    0.371
object  1.0 vs 0.25 fps   29   19    0.194
object  1.0 vs 0.5 fps   16   10    0.327
object  1.0 vs 2 fps     16   16    0.860

[二分对比] action vs 其余（R@1），1 fps 对比其他抽帧率
组           n 对比                b    c        p
action    375 1.0 vs 0.25 fps   23   16    0.337
action    375 1.0 vs 0.5 fps   20   12    0.216
action    375 1.0 vs 2 fps     14   11    0.689
rest      625 1.0 vs 0.25 fps   34   26    0.366
rest      625 1.0 vs 0.5 fps   17   13    0.584
rest      625 1.0 vs 2 fps     17   20    0.742


### 2026-11-18 创新点 B / C：预算选择与成本主导项（纯推导，不新增实验）
脚本：`code/analyze_budget_scale.py`（只读 `results/results.csv`）
输出：`results/derived_budget_scale.csv`
假设：机器时间 50 CNY/h（与 §IV 一致）；内存 25 CNY/GB（2026 年 DDR4 市场价量级，报告需注明是假设）。

**B. 预算约束下的配置选择**
增量成本与增量精度（MSR-VTT，n=1000，max 聚合）：

| 从 → 到 | 建库时间增量 | R@1 增量 | 每 pp 的秒数 |
|---|---|---|---|
| 0.25 → 0.5 fps | +266.6 s | +0.3 pp | 889 s/pp |
| 0.5 → 1.0 fps | +491.7 s | +1.2 pp | 410 s/pp |
| 1.0 → 2.0 fps | +724.0 s | +0.0 pp | ∞ |

- 最便宜配置 0.25 fps：277.5 s / R@1 30.7%；R@1 最高配置 1.0 fps：1035.8 s / 32.2%；
  多花 758.3 s 只换来 1.5 pp（正好 1 个标准误）。
- 结论：低于 1 fps 时钱能买到小但真实的东西（410–890 s 换 1 pp）；高于 1 fps 汇率归零。
- 推论（这是 B 的价值）：预算若已超过约 1040 s/千段，应把增量投入**第二阶段重排**
  （对应 34.3 pp 空间），而不是继续加抽帧率（对应 0.0 pp）；重排只需对 top-10 打分，
  不需要重新编码整个语料。

**C. 算力 vs 存储（每帧）**
- 算力：65 ms/帧 × 50 CNY/h ÷ 3600 = 9.03e-4 CNY/帧
- 存储：2048 B/帧 × 25 CNY/GB = 5.12e-5 CNY/帧
- 比值：**算力是存储的 17.6 倍**，且两者都与帧数线性，**比值不随规模变化**
- 因此"规模变大后存储必然主导"在本设计中**不成立**（依据实测数字）。
  要让存储与算力持平，内存价格需达到 **441 CNY/GB**（现价假设的 17.6 倍）。
- 存储真正限制的不是钱，而是**装不装得下**（内存墙）：
  15.8 GB ÷ 2048 B = **7.7e6 帧** ≈ 2,140 小时视频（1 fps）；
  到墙时建库算力 ≈ 139 h ≈ 6,965 CNY，而同样的向量只值约 395 CNY 内存。
- 结论：本设计的下一个瓶颈是 **~1e7 帧处的内存容量**，不是存储成本；
  过了这道墙就必须压缩（PQ），这正是 LOVO 在 3e7 实体上做量化、而我们能继续用 float32 的原因。

口径说明：
- 与 `notes/context.md` §2.2 原计划（"随 N 增大，主导项从算力迁移到存储，求临界规模 N*"）
  **不一致**：实测算力/存储 = 17.6 且与 N 无关，不存在价格意义上的 N*。
  已按实测改写为"内存墙 + 表示冗余（每帧向量数 K）"的说法，报告正文按改写后的版本写。
- 上述数字均为**推导值**（由 results.csv 与两个假设算出），与实测值分开存放于
  `results/derived_budget_scale.csv`，正文引用时必须说明是推导。


### 2026-11-19 两处加固：4 fps 补点 + C 的双价格点

**1) 4 fps 补点（进行中）**
目的：把"1 fps 以上换不来精度"从 2 个点（1.0 / 2.0 fps）加固到 3 个点（1 / 2 / 4 fps）。
命令：`python code/build_index.py --fps 4 --out-dir data\index_msrvtt_fps4`
预期帧数：1000 段 × 4 fps，约 6.4 万帧（对照 1 fps 的 16,036 帧、2 fps 的 22,412 帧）。
评测：`python code/evaluate.py --protocol msrvtt --index-dir data\index_msrvtt_fps4 --topk 10
      --save results/msrvtt_vitb32_fps4.json`
读数纪律：结果出来后同时写 `results/results.csv` 与本节，两处一致才算数；
若 4 fps 的 R@1 仍与 1 fps 相差 ≤1.5 pp，则结论写成"1–4 fps 之间测不出增益"；
若 4 fps 明显上升，则把结论改成"平台在 1–2 fps"，不得继续沿用旧句式。

**2) C 改为双价格点（已完成）**
原因：审稿人可能质疑"25 CNY/GB 是随便假设的"。改为同时给出两种放置方式：
- 内存驻留：25 CNY/GB → 存储 5.12e-5 CNY/帧 → 算力是存储的 **17.6 倍**
- 放固态盘：0.5 CNY/GB → 存储 1.02e-6 CNY/帧 → 算力是存储的 **882 倍**
结论方向不变且更稳：无论放内存还是放磁盘，算力都远高于存储成本；存储真正起作用的方式是
"装不下"（15.8 GB = 7.7e6 帧 ≈ 2,140 小时视频），而不是"更贵"。
推导值已更新到 `results/derived_budget_scale.csv`（新增 storage_cny_per_frame_ssd、
compute_over_storage_ssd 两行）。

**3) B 的措辞收紧（已完成）**
原文"above 1 fps it cannot be bought at all"改为"above 1 fps the exchange rate is zero over
the range we measured"，避免把有限区间的测量说成普适规律。


#### 4 fps 补点的坑：`--max-frames 30` 会把 4 fps 截断成 2 fps（2026-11-19）
第一次启动 `--fps 4` 时用的是默认 `--max-frames 30`。查看 `sample_frames()` 发现：
`indices = list(range(0, total, step))[:max_frames]`，即"先按帧率取索引，再截断到 30 帧"。
15 秒的片段在 4 fps 下需要 60 帧，截断成 30 帧后**与 2 fps 完全等价**，这样跑出来的点毫无意义。
处理：终止进程、删除半成品目录 `data/index_msrvtt_fps4`，改用
`--fps 4 --max-frames 60` 重新建库。

顺带发现一个必须写进报告的方法学细节：
**2.0 fps 那个点其实被 30 帧上限截断了一部分**（长于 15 秒的片段需要 30 帧以上）。
所以实测 2 fps 的帧数只有 22,412（1 fps 的 1.40 倍），而不是理论上的 2 倍。
方向对我们有利：真实的 2 fps 只会更贵、精度不变，所以"加帧率不值"的结论更稳。
但报告里必须如实说明这个上限，否则读者会按 2 倍去核算成本、发现对不上。
（正文处理：在 V-C 或 Setup 加一句 "the 30-frame cap binds above 1 fps for clips longer than
15 s, so the 2.0 fps point understates the cost of that rate"。）

4 fps 参数（本次实际使用）：
`--fps 4 --max-frames 60 --out-dir data\index_msrvtt_fps4`，预期约 6 万帧、建库约 80 分钟。


### 2026-11-19 §VI-C 重写：50 CNY/h 这个假设站不住，改成价格参数化

**自查发现的问题（重要）**
原稿把机时价格假定为 50 CNY/h，并据此写 "算力是存储的 17.6 倍"。
但是：同一篇论文 §IV 里，按 45 W、0.55 CNY/kWh 算出的电费是
**每机时约 0.03 CNY**（3,600 帧 / 7.4 MB / 0.0036 kWh → 0.2 分钱）。
4 vCPU 云实例的真实价格也就在 1 CNY/h 量级。
也就是说，50 CNY/h 比论文自己用的电费高 1,600 倍，比云 CPU 高约 50 倍——
它实际上是 GPU 租用价，而我们全篇都强调"CPU-only、没有 GPU"。
审稿人只要把这两处并排看，就会问"你的机时凭什么值 50 CNY/h"，而这个问题在原稿里没有答案。

**更麻烦的是方向会翻**
存储成本与 P 无关，算力成本 = 1.8056e-5 × P（CNY/帧）。于是：
- 交叉点 P* = 5.12e-5 / 1.8056e-5 = **2.84 CNY/h**（内存 25 CNY/GB）
-  SSD 口径：P* = **0.057 CNY/h**
- P = 0.03（仅电费）：**存储比算力贵 94.5 倍**
- P = 1（云 CPU）：存储贵 2.84 倍
- P = 50（GPU 租用）：算力贵 17.6 倍
所以"算力主导"只在 GPU 级价格下成立；按论文自己的电费口径，结论正好相反。

**处理方式（已改正文）**
不再给单一比值，改成把机时价格 P 当变量：
- 算力 $1.8\times10^{-5}P$ CNY/帧 vs 内存 $5.1\times10^{-5}$ / SSD $1.0\times10^{-6}$
- 明写交叉点：内存 2.8 CNY/h、SSD 0.06 CNY/h
- 给出两端锚点：仅电费（0.03）时存储贵两个数量级；GPU 级（几十 CNY/h）时算力贵一个数量级
- 保留下来的**与价格无关**的结论：两项都与 N 线性 → 不存在随规模出现的交叉点；
  真正随 N 增长的约束是**内存容量墙 7.7e6 帧**，所以补救手段是量化索引（LOVO/FAISS 的做法）

同时 §IV-E 的 GPU 回收点补了一句敏感性：交叉点与机时价格成反比，
按 1 CNY/h 算是 5.6–7.2e8 帧（约 2e5 小时视频），即"买 GPU 只在机时很贵时才划算"；
Limitations 里"应当上 GPU"改成"GPU 或近似索引成为必需"。
§IV-E 的 50 CNY/h 现在明确标注为"GPU 租用价（高锚点）"，两处口径不再打架。

**落盘**
`results/derived_budget_scale.csv` 新增：compute_cny_per_frame_basis=50、
compute_cny_per_frame_per_cny_per_hour=1.80556e-5、crossover_machine_price_ram=2.84、
crossover_machine_price_ssd=0.0567、compute_over_storage_ram_at_0.03=0.0106、
storage_over_compute_ram_at_0.03=94.5、compute_over_storage_ram_at_1=0.353。
`code/analyze_budget_scale.py` 增加机时价格敏感性打印。
旧行（compute_over_storage_ram=17.63 等）保留，含义是"P=50 时"，新增 basis 行已说明。
重编译：6 页、0 Overfull、`audit_numbers.py` 0 处 DIFF。


### 2026-11-19 【更正】抽帧率标签错了：2.0 实际是 1.5，4.0 实际是 3.0

**怎么发现的**
4 fps 建库结束后核对 meta，发现每帧时间戳间隔是 0.33 s（= 3.03 fps），不是 0.25 s。
回头查所有索引的 meta 时间戳，并直接读源视频的 `CAP_PROP_FPS`（抽 40 段，全部 = 3.0）：

| 目录 | 标签 | 实测间隔 | 实际帧率 | 帧数 |
|---|---|---|---|---|
| index_msrvtt_fps0.25 | 0.25 | 4.00 s | 0.25 | 4,344 |
| index_msrvtt_fps0.5 | 0.5 | 2.00 s | 0.50 | 8,285 |
| index_msrvtt | 1.0 | 1.00 s | 1.00 | 16,036 |
| index_msrvtt_fps2 | **2.0** | 0.67 s | **1.49** | 22,412 |
| index_msrvtt_fps4 | **4.0** | 0.33 s | **3.03** | 44,435 |

原因：**数据集里的 mp4 本身已经是 3 fps 预抽好的**，而 `build_index.py` 用
`step = round(src_fps / target_fps)`：目标 2 fps → step 2 → 1.5 fps；目标 4 fps → step 1
→ 已经取不到更多帧，实际 3 fps（这是数据的物理上限）。

**被推翻的旧说法（保留旧记录，此处更正）**
本日早些时候的条目里写过"2.0 fps 被 30 帧上限截断，所以帧数只有 1.40×"。
**这是错的**：1.40× 的真实原因是帧率本身只有 1.5（1.5/1.0 = 1.5，扣除 cap 后 1.40）。
旧说法已从正文删除，不再引用。

**3 fps 的实测结果（新）**
- 帧数 44,435，索引 91.00 MB（44,435 × 512 × 4 B = 91.0 MB ✓）
- R@1 32.2% / R@5 55.9% / R@10 65.6%（1 fps 为 32.2 / 55.6 / 66.5）
- 结论更强：**3 fps 已经用上了数据集提供的每一帧，R@1 仍然没有变化**

**3 fps 的耗时怎么来的（必须写明）**
这次建库的总耗时 49,116.4 s 不可用：日志里 320→330 段之间跳了 19,453 s、540→550 段之间
跳了 17,478 s，是电脑休眠，不是计算。取两段干净区间：
- 0→320 段：14,384 帧 / 1,145 s
- 550→1000 段：19,952 帧 / 1,592 s（末段 24,483→44,435 帧，47,523→49,115 s）
合计 34,336 帧 / 2,737 s = **79.7 ms/帧**，两段各自算也都是 79.6 / 79.8 ms，一致。
于是全量折算 44,435 × 79.7 ms = **3,542 s**，作为该行的 build_seconds 记入 results.csv，
并在 note 字段注明是派生值。若日后要纯实测值，重跑一次约 60 分钟（需禁用睡眠）。

**已同步**
- `results/results.csv`：2.0 行改为 1.5（note 说明原因），新增 3.0 行
- `results/query_types.csv`：fps 2 → 1.5（同一批索引）
- `report/main.tex`：Table IV 五档（3.0/1.5/1.0/0.5/0.25）、Table V 表头 1.5、
  §V-C 重写、Fig.2 图注、Abstract「0.25–3.0 fps / 13×」、Conclusion、
  §VI-B「1→1.5 fps 724 s；1.5→3.0 fps 1,782 s」
- `results/fig_cost_accuracy.png` 重新生成（5 个点，文字包围盒无重叠）
- `code/audit_numbers.py`：期望值改为 5 档，并新增 3 fps 相关断言；当前 0 处 DIFF


### 2026-11-19 3 fps 重跑（干净实测，替换掉上面的 3,542 派生值）

**为什么重跑**：上一版 3 fps 用了 `--max-frames 60`，等于把超过 20 s 的片段截断（帧数只有
1 fps 的 2.77 倍，正文只能说“2.8 倍而不是 3 倍”），而且耗时被休眠污染过，只能派生。
这次用 `--max-frames 90`（3 fps × 30 s）重跑，全程没休眠。

**命令**
```
python -u code/build_index.py --fps 3 --max-frames 90 \
    --data-dir data/msrvtt/video --out-dir data/index_msrvtt_fps3_full
```

**实测结果（这就是现在 results.csv / Table IV 里的那行）**
| 项 | 值 |
|---|---|
| 帧数 | 47,121（1,000 段，每段 32–90 帧） |
| 建库耗时 | 3,927.1 s = 65.5 min |
| ms/帧 | 83.3（日志全程 77–90，无跳变；总 100 条进度行，无 >200 ms/帧 区间） |
| 索引 | 96.50 MB（47,121 × 512 × 4 B = 96.5 MB ✓） |
| R@1 / R@5 / R@10 | 32.4% / 56.1% / 65.2% |

**配对检验（这是新的证据，比我原来靠“一个标准误”讲更硬）**
| 比较 | R@1 p | R@10 p |
|---|---|---|
| 1.0 vs 3.0 fps | 0.883 | 0.106 |
| 0.25 vs 3.0 fps | 0.122 | 0.338 |
最小 p = 0.106 ⇒ “整个 12× 区间内没有显著差异”在 α=0.05 下成立。

**两个必须如实说明的细节（写进了 Table IV 图注和 §V-C）**
1. **90 帧上限确实碰到过 12 段**：12 段视频长 30.3–30.7 s（源 91–92 帧），被截到 90 帧。
   少掉 24 帧，占索引 47,121 帧的 0.05%。所以只能说“几乎每一帧”，不能说“每一帧”。
2. **帧数比是 2.9× 而不是 3×，原因是取整**：1 fps 每段保留 ⌈N/3⌉ 帧。拿 3 fps 的每段帧数
   按 ⌈N/3⌉ 求和 = 16,036，与 1 fps 索引实测完全一致（1,000 段零误差）。与 cap 无关。

**连带改动**
- `results/results.csv`：3.0 行全量替换（47,121 / 3,927.1 / 83.3 / 96.50 / 32.4-56.1-65.2），
  note 去掉 DERIVED，改为说明 12 段截断
- `results/msrvtt_vitb32_fps3.json`：用新索引重跑 evaluate 覆盖
- `report/main.tex`：Table IV 行与图注、§V-C（14×、3.8×、2.9×、p=0.88/0.11）、
  Abstract（14×、1.7 pp）、Contributions（$p=0.12$）、§VI-B（1.5→3.0 = 2,167 s / +0.2 pp，
  “交换率”从“零”改为“比最优段差一个数量级”）、Conclusion（“nearly four times”）
- `results/fig_cost_accuracy.png` + `report/` 副本：重新生成（5 点，成本跨度 14.2×，R@1 跨度 1.7 pp，
  文字包围盒无重叠）
- `code/audit_numbers.py`：新增 3 fps vs 1 fps 成本、2,167 s 两条断言；当前 **0 处 DIFF**
- 目录：`data/index_msrvtt_fps3`（44,435 帧旧版）删除，新索引从 `_full` 改名为该名
- `README.md`：`--fps 3` 命令改为 `--max-frames 90`

**被本条取代的旧记录**：上面那条“3 fps 耗时 3,542 s（79.7 ms/帧 派生）”作废。
保留原记录只作为过程痕迹，正文与 CSV 不再引用该派生值。


### 2026-11-19 补记：R@10 在 0.25 fps 上是真的显著（不能笼统说“全区间无差异”）

重跑后顺手把 1.0 与 0.25 fps 也做了配对检验（同一批查询，McNemar）：

| 比较 | R@1 | R@10 |
|---|---|---|
| 1.0 vs 0.25 fps | p=0.159（不显著） | **p=0.027（显著，66.5% → 64.1%，−2.4 pp）** |

所以之前正文里那句“整个区间都没有显著差异”只能限定到 R@1。已改为：
- §V-C：R@1 不显著（p=0.16）；R@10 在 0.25 fps 上显著下降 2.4 pp（p=0.03）
- Table IV 图注：同上，并保留“3 fps 的 12 段 30 s 以上片段被截到 90 帧（0.05%）”
- Fig.2 图注：只对 Recall@1 说“无法区分（p≥0.11）”

净效果：结论其实更强——“1 fps 是拐点”现在有两层证据（R@1 平、R@10 在更低的 0.25 fps 才开始掉）。

**页数**：这次新增内容一度把 PDF 撑到 7 页（第 7 页只有 3 条参考文献）；压缩 §V-C、
Table IV 与 Fig.2 图注后回到 **6 页**，0 错误、0 Overfull，`audit_numbers.py` 仍 0 DIFF。


### 2026-11-19 全篇自查：多重比较校正 + 五处数字/引用修正

**1. 新增 `code/analyze_significance.py` → `results/significance.csv`**
正文引用的 5 个 p 值原来散在临时脚本里，别人无法复现。现在一条命令输出全部 10 对、
两个指标（R@1 / R@10）的 McNemar 检验，并给出 Holm 校正值。

| 指标 | 最小未校正 p | 最小 Holm p | 结论 |
|---|---|---|---|
| R@1 | 0.122（0.25 vs 3.0） | 1.000 | 全平，无任何显著对 |
| R@10 | 0.017（0.25 vs 1.5） | 0.172 | 校正后无一显著 |

**R@10 的三对未校正显著（0.25 vs 1.0 p=0.027；0.25 vs 1.5 p=0.017；1.5 vs 3.0 p=0.030）
方向互相矛盾**（0.25 比 1.0 差，但 3.0 又比 1.5 差），且 10 次比较下应预期 0.5 个假阳性。
所以正文原来那句“R@10 在 0.25 fps 显著下降”作废，改为“校正后无任何差异，最大未校正变化
2.4 pp 出现在 R@10 的 0.25 vs 1.0”。这比原来的说法更稳。

**2. 片段平均时长错了：15.0 s → 15.7 s**
从 3 fps 索引的 meta 实测：1,000 段共 15,703.8 s = 4.36 h，平均 15.704 s（中位 13.66 s，
最长 30.00 s）。原 §V-B 写 15.0 s，并据此写“16.0 帧 = 15.0 s × 1 fps”，这个等式本身就不成立。
已改为 15.7 s，并把 30 fps 的对照值从 480 改为约 471（15.7 × 30）。

**3. 删除无出处的“published zero-shot range ≈30–33%”（正文 3 处 + README 1 处）**
仓库里没有任何文献支持这个区间；审稿人一问“哪篇”就答不上来。改为可自证的说法：
Recall@1 32.2% 比 0.1% 随机水平高 32 个百分点，且四个独立量（帧数、隐含命中数、索引大小、
随机基线）互相吻合，因此管线实现正确。

**4. GPU 回收点：1 CNY/h 时 5.6–7.2e8 → 5.7–7.3e8 帧，小时数 2e5 → 1.6–2.0e5**
按 13,000 CNY / 50× 加速 / 65–84 ms 重算：5.685e8 与 7.347e8。

**5. 其他一致性修正**
- 硬编码 “Section IV / IV-E” 改为 `\ref{sec:cost}` / `\ref{sec:boundary}`
- `results/msrvtt_vitb32_fps2.json` → `msrvtt_vitb32_fps1.5.json`
- `code/analyze_budget_scale.py`：原先把 R@1 最高的 3 fps 当基准，导致“推荐 0.5 fps”与正文冲突；
  改为以 1 fps 为基准点报增量（现在输出 267/492/724/2167 s 与 +0.3/+1.2/0.0/+0.2 pp，与 §VI-B 一致）
- “encoder 比解码贵 6–7 倍” → “约 7 倍”（实测 7.2 与 6.5）
- 压缩了 Introduction / §V-C / Table IV 图注 / §VI-C 结尾，页数从 7 页回到 **6 页**

**当前闸门**：`audit_numbers.py` 0 DIFF、0 SKIP；PDF 6 页、0 错误、0 Overfull。


### 2026-11-20 找回“与已发表结果一致”的句子（这次有出处）

**出处**：L. Ventura, C. Schmid, G. Varol, "Learning Text-to-Video Retrieval from Image
Captioning," arXiv:2404.17498, 2024, Table 5。同一张表里有两行零样本 CLIP ViT-B/32
在 MSR-VTT 1k（单句查询、T2V）上的 R@1：

| 方法（该论文 Table 5） | Backbone | MSR-VTT R@1 / R@5 |
|---|---|---|
| CLIP + query-scoring（作者实现） | ViT-B/32 | 30.7 / 54.0 |
| CLIP4Clip 零样本 | ViT-B/32 | 31.2 / 53.7 |

我们的 32.2% 比这两个数高 1.0–1.5 pp，正好是一个标准误（1.5 pp）。协议差异也写明：
他们在 10 帧上做 mean/query-scoring 池化，我们在约 16 帧上做 max 池化。

**改了哪三处**
- Abstract：`far above the 0.1% chance level` → `within 1.5 pp of the 30.7–31.2% reported for
  zero-shot CLIP ViT-B/32 on this split [ref]`
- Introduction：删除“不与已发表数字比较”，改为“与 30.7–31.2% 一致，在一个标准误内，协议有差异”
- §III-B：同样改为与这两个数比较

**页数**：新增 1 条参考文献 + 3 处文字后一度 7 页；压缩 §IV-A（五项成本）与 §III-C
（刻意省略的组件）后回到 **6 页**，0 错误 0 Overfull。

**闸门**：`audit_numbers.py` 仍 0 DIFF；9 条文献全部被引用且都存在；PDF 9 个编号全部出现。
