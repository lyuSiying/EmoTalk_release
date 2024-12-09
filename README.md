本项目用于记录语音识别大作业所有工作，包括部署环境，修改原仓库代码以适应不同环境，读论文学习方法等等，
当然最重要的是避免最终结课前网速太慢传不到老师仓库（哭）
文件结构和组成如下：
  originalCodeBackup：fork原仓库
  revisedDesign：我们的工作汇总
    --EmoTalk：修改后的可运行项目，但是需要在hugging face手动下载wav2vec2-large-xlsr-53-english和wav2vec-english-speech-emotion-recognition模型并存放到models文件夹，还需要手动下载EmoTalk.pth并存放到pretrain_model,其他参考环境部署教程
    --test_for_final：测试老师测试集的各项评价指标
    --train：将原仓库的推理模型改为训练
    其他工作会不定时更新，readme会同步更新，最终不知道老师仓库要交什么，总之其他工作如果无需提交可以直接给链接跳转到这个仓库（doge）

# 语音识别大作业记录

本项目用于记录语音识别大作业的所有工作，包括以下内容：
- **部署环境**：搭建和配置运行环境。
- **代码修改**：对原仓库代码进行修改以适应不同环境。
- **论文学习**：学习相关方法并进行记录。
- **备份与提交**：避免结课前因网速问题无法将代码上传至老师的仓库（哭）。

## 文件结构和组成
```plaintext
originalCodeBackup/   # fork 原始仓库
revisedDesign/        # 我们的工作汇总
  ├── EmoTalk/          # 修改后的可运行项目
  ├── test_for_final/   # 测试老师测试集的各项评价指标
  ├── train/            # 将原仓库的推理模型改为训练模型
其他工作/              # 不定时更新

1. originalCodeBackup
这是我们 fork 的原始仓库，保留为备份。
2. revisedDesign
我们的主要工作汇总，包括以下子模块：

2.1 EmoTalk
修改后的可运行项目。
注意事项：
需要手动下载以下模型并放置到指定目录：
wav2vec2-large-xlsr-53-english 和 wav2vec-english-speech-emotion-recognition，下载后存放至 models/ 文件夹。
EmoTalk.pth，下载后存放至 pretrain_model/ 文件夹。
其他具体信息参考环境部署教程。
2.2 test_for_final
用于测试老师提供的测试集，计算各项评价指标。
2.3 train
将原仓库的推理模型修改为训练模型，实现自定义训练功能。
3. 其他工作
其他内容会不定时更新，README.md 将同步更新。
如果最终老师的仓库要求不需要提交某些工作，可以直接通过链接跳转到本项目。