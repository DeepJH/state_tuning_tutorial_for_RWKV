# 🚀 State Tuning 教程 - RWKV模型微调指南

## 📖 什么是State Tuning?

RWKV 是纯 RNN 架构，因此能够实现 Transformer 难以做到的功能。由于 RNN 有固定大小的 state，微调 RWKV 的初始 state 相当于最彻底的 prompt tuning，甚至可用于对齐（alignment），因为其迁移能力非常强。

## 💡 如何理解 State Tuning?

简单来说，State Tuning 就是修改模型的"临时记忆"，无需输入提示词和上下文，直接达到相同或更好的状态以供使用。

- ✅ **不修改模型本身**，微调节省资源
- ✅ **更底层操作**，微调更加精确
- ✅ **适用于强化学习（RL）**，可以"插入"特定临时记忆，"指导"模型学习

# 🛠️ 环境准备

本文的 State Tuning 方法基于 RWKV 社区微调项目 [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT)。

开始之前，请确保你拥有一个 Linux 工作区，以及支持 CUDA 的 NVIDIA 显卡。

## 📊 State Tuning 的显存需求

| 模型参数 | bf16 | int8 | nf4 |
| :--: | :--: | :--: | :--: |
| RWKV7-0.1B | 2.6GB GPU | 2.4GB GPU | 2.5GB GPU |
| RWKV7-0.4B | 3.1GB GPU | 2.9GB GPU | 2.8GB GPU |
| RWKV7-1.5B | 5.3GB GPU | 4.1GB GPU | 3.7GB GPU |
| RWKV7-2.9B | 8.2GB GPU | 5.7GB GPU | 4.7GB GPU |

### 🔍 精度对比说明：

| 维度 | bf16 | int8 | nf4 |
| :--: | :--: | :--: | :--: |
| 占用显存 | 最高 (100%) | 中 (50%) | 最低 (~25% - 30%) |
| 精度保持 | 极佳（原生） | 良好（可能有损） | 优秀（针对 4-bit 而言） |
| 计算速度 | 快（原生支持） | 快（需硬件支持） | 较慢（推理时需反量化回 BF16） |
| 硬件要求 | A100 / 30系列及以上 | 普遍支持 | 普遍支持（需 bitsandbytes 库） |

## 📥 准备训练数据

微调数据要求：[准备微调数据](https://www.rwkv.cn/tutorials/advanced/Fine-Tune/FT-Dataset)

本次示例将实现[视频](https://space.bilibili.com/1459296148/lists/7256133?type=season)中的NEGA（Native English Grammar Assistant）
![NEGAS](./images/NEGAS.png)
1. 使用[浏览器插件](https://chromewebstore.google.com/detail/%E7%88%B1%E5%93%94%E5%93%94-bilibili-b%E7%AB%99%E5%8A%A9%E6%89%8B/nabiilbmnbbgdbhkgdghdbcldakfeofd)，批量下载B站上所有的NEGA对话视频

2. 使用[buzz](https://github.com/chidiwilliams/buzz)，批量将视频中的对话提取为SRT字幕文件  

3. 使用[`nigga_to_NEGA_and_srt_to_jsonl.py`](./files/nigga_to_NEGA_and_srt_to_jsonl.py)将SRT字幕文件转换合成为jsonl数据集，并纠正部分语音识别错误，得到[`NEGA.jsonl`](files/NEGA.jsonl)
![NEGA.jsonl.png](./images/NEGA.jsonl.png)
4. 在vscode中使用查找替换功能，手动纠正语音转文字的错误

5. 由于数据量非常少，只有17行，还需要合成一些数据。使用[`prompt_generate_generate_NEGA_explain.md`](./files/prompt_generate_generate_NEGA_explain.md)、[`prompt_generate_generate.md`](./files/prompt_generate_generate.md)、[`prompt_generate_NEGA_explain.md`](./files/prompt_generate_NEGA_explain.md)、[`prompt_generate.md`](./files/prompt_generate.md)提示词，让gemini生成更多数据。反复多次。

6. 得到最终数据集[`NEGA_PLUS.jsonl`](files/NEGA_PLUS.jsonl)
![NEGA_PLUS.jsonl.png](./images/NEGA_PLUS.jsonl.png)

## 🛠️ 配置训练环境

配置训练环境可参考[配置训练环境](https://www.rwkv.cn/tutorials/advanced/Fine-Tune/FT-Environment)。

使用conda创建多个虚拟环境，各自安装依赖，防止依赖相互冲突，主要分为：微调（RWKV-PEFT）、推理（RWKV-Runner）、数据整理。

## 📥 克隆仓库并安装依赖

进入虚拟环境，使用 git 命令克隆 RWKV-PEFT 仓库：
```
git clone https://github.com/JL-er/RWKV-PEFT.git
```
如果 GitHub 无法连接，请使用以下国内仓库：
```
git clone https://gitee.com/rwkv-vibe/RWKV-PEFT.git
```
克隆完成后，使用 cd RWKV-PEFT 命令进入 RWKV-PEFT 目录。并运行以下命令，安装项目所需依赖：
```
cd RWKV-PEFT
pip install -r requirements.txt
```
## 📥 下载 rwkv7 模型
在[这里](https://huggingface.co/BlinkDL/rwkv7-g1/tree/main)下载RWKV模型。参考[State tuning 的显存需求](#State-tuning-的显存需求)  
鉴于我的显存为8GB，选择下载rwkv7-g1c-1.5b-20260110-ctx8192.pth。	
# 🚀 训练
## 🔧 调整训练参数
使用任意文本编辑器编辑`RWKV-PEFT/scripts/state tuning.sh`文件，修改训练参数，进而控制微调的训练过程和训练效果：
```sh
load_model="/home/deepjh/workspace/RWKV_LAB/models/rwkv7-g1c-1.5b-20260110-ctx8192.pth"
proj_dir="/home/deepjh/workspace/RWKV_LAB/project_tuning/RWKV-PEFT/state-models"
data_file="/home/deepjh/workspace/RWKV_LAB/dataset/NEGA_PLUS.jsonl"
#/home/rwkv/JL/data/roleplay
n_layer=24
n_embd=2048

micro_bsz=8
epoch_save=1
epoch_steps=2560
ctx_len=512

python train.py --load_model $load_model \
--proj_dir $proj_dir --data_file $data_file \
--vocab_size 65536 \
--data_type jsonl \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz \
--epoch_steps $epoch_steps --epoch_count 4 --epoch_save $epoch_save \
--lr_init 5e-5 --lr_final 5e-6 \
--accelerator gpu --precision bf16 \
--devices 1 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--peft state --op fla \
--wandb NEGA

```
## 🚀 开始训练
在 RWKV-PEFT 目录下，运行`sh scripts/state/ tuning.sh`命令，开始 state tuning 。  
正常开始训练后，应当是如下画面，然后等待训练完成：
![training](./images/training_NEGA.png )

# 📊 结果
## 📈 wandb
- 配置wandb。注册wandb，在命令行需要时填入api key。
- 在RWKV-PEFT目录下执行`wandb sync wandb/offline-*`，同步本地所有数据。或者执行`wandb sync wandb/latest-run`仅同步最近一次运行。
- 查看训练结果。以下为loss曲线。
![loss](./images/loss3.png)
## 📖 运行效果
- 使用[RWKV-Runner](https://github.com/josStorer/RWKV-Runner)运行，按照官方的教程完成部署。  
- 将rwkv7-g1c-1.5b-20260110-ctx8192.pth放入对应model目录，将训练好的state模型rwkv-4.pth放入state-models目录。  
- 在config界面，新建一个配置，调整相关设置，点击右下角启动，如图
![config](./images/config2.png)
- 回到chat即可聊天，效果如图：
![chat](./images/chat_0_2.png)
