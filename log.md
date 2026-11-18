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
