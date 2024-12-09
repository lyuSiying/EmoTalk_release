本项目用于记录语音识别大作业所有工作，包括部署环境，修改原仓库代码以适应不同环境，读论文学习方法等等，
当然最重要的是避免最终结课前网速太慢传不到老师仓库（哭）
文件结构和组成如下：
  originalCodeBackup：fork原仓库
  revisedDesign：我们的工作汇总
    --EmoTalk：修改后的可运行项目，但是需要在hugging face手动下载wav2vec2-large-xlsr-53-english和wav2vec-english-speech-emotion-recognition模型并存放到models文件夹，还需要手动下载EmoTalk.pth并存放到pretrain_model,其他参考环境部署教程
    --test_for_final：测试老师测试集的各项评价指标
    --train：将原仓库的推理模型改为训练
    其他工作会不定时更新，readme会同步更新，最终不知道老师仓库要交什么，总之其他工作如果无需提交可以直接给链接跳转到这个仓库（doge）
