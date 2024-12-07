import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
from model import EmoTalk


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_path, sr=16000):
        self.data = []
        with open(data_path, 'r') as f:
            for line in f:
                audio_path, blendshape_path = line.strip().split(',')
                self.data.append((audio_path.strip('"'), blendshape_path.strip('"')))
        self.sr = sr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        audio_path, blendshape_path = self.data[index]
        # 修复路径问题，移除多余的引号
        audio_path = audio_path.strip('"')
        blendshape_path = blendshape_path.strip('"')

        # 加载音频
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr)
            print(f"Loaded audio from {audio_path}, shape: {audio.shape}")

        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return None  # 如果音频加载失败，返回 None

        # 确保音频是 2D 张量：[1, time_steps]
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # [time_steps] -> [1, time_steps]
        print(f"Audio tensor shape after unsqueeze: {audio_tensor.shape}")

        # 如果音频数据是多通道（例如立体声），则确保它的形状为 [1, num_channels, time_steps]
        if len(audio_tensor.shape) == 2:  # 可能是多通道（如立体声）
            audio_tensor = audio_tensor.unsqueeze(0)  # [num_channels, time_steps] -> [1, num_channels, time_steps]

        # 加载目标 blendshape 系数
        try:
            blendshape_gt = np.load(blendshape_path)
        except Exception as e:
            print(f"Error loading blendshape file {blendshape_path}: {e}")
            return None  # 如果 blendshape 加载失败，返回 None



        blendshape_tensor = torch.FloatTensor(blendshape_gt)
        return audio_tensor, blendshape_tensor


def train(args):
    # 初始化模型
    model = EmoTalk(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_cross = nn.MSELoss()  # 用于交叉重构损失
    criterion_self = nn.MSELoss()  # 用于自我重构损失
    criterion_velocity = nn.MSELoss()  # 用于速度损失

    # 加载数据集
    train_dataset = CustomDataset(args.train_data_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in train_loader:
            if batch is None:
                continue  # 如果返回的批次为空，则跳过该批次

            audio, blendshape_gt = batch
            audio, blendshape_gt = audio.to(args.device), blendshape_gt.to(args.device)

            frame_num = blendshape_gt.shape[1]

            # 使用 processor 和 feature_extractor 来提取内容和情感特征
            inputs12 = model.processor(torch.squeeze(audio), sampling_rate=16000, return_tensors="pt",
                                       padding="longest").input_values.to(args.device)

            # 打印输入维度
            print(f"Shape of inputs from processor: {inputs12.shape}")  # 打印输入的维度

            # 提取内容特征
            hidden_states_cont1 = model.audio_encoder_cont(inputs12, frame_num=frame_num).last_hidden_state
            print(f"Shape of hidden_states_cont1: {hidden_states_cont1.shape}")  # 打印提取的内容特征的维度

            # 提取情感特征
            inputs_emo = model.feature_extractor(torch.squeeze(audio), sampling_rate=16000, padding=True,
                                                 return_tensors="pt").input_values.to(args.device)
            print(f"Shape of inputs_emo: {inputs_emo.shape}")  # 打印情感特征输入的维度

            output_emo = model.audio_encoder_emo(inputs_emo, frame_num=frame_num)
            hidden_states_emo = output_emo.hidden_states
            print(f"Shape of hidden_states_emo: {hidden_states_emo.shape}")  # 打印情感特征提取的维度

            hidden_states_emo_832 = model.audio_feature_map_emo(hidden_states_emo)
            hidden_states_emo_256 = model.relu(model.audio_feature_map_emo2(hidden_states_emo_832))

            # 情绪解缠生成情感强度和人物风格
            onehot_level = model.one_hot_level[output_emo.logits.argmax(dim=-1).cpu().numpy()]
            onehot_level = torch.from_numpy(onehot_level).to(args.device).float()
            print(f"Shape of onehot_level: {onehot_level.shape}")  # 打印 onehot_level 的维度

            # 假设随机生成人物风格，可以根据实际需求调整
            onehot_person = torch.randn((audio.shape[0], 24))  # 随机生成人物风格
            onehot_person = torch.from_numpy(onehot_person).to(args.device).float()
            print(f"Shape of onehot_person: {onehot_person.shape}")  # 打印 onehot_person 的维度

            # 嵌入情感强度和人物风格
            obj_embedding_level = model.obj_vector_level(onehot_level).unsqueeze(1).repeat(1, frame_num, 1)
            obj_embedding_person = model.obj_vector_person(onehot_person).unsqueeze(1).repeat(1, frame_num, 1)

            # 拼接输入特征
            hidden_states = torch.cat(
                [hidden_states_cont1, hidden_states_emo_256, obj_embedding_level, obj_embedding_person], dim=2)
            print(f"Shape of hidden_states after concatenation: {hidden_states.shape}")  # 打印拼接后的 hidden_states 维度

            # 前向传播
            prediction = model.bs_map_r(hidden_states)
            print(f"Shape of prediction: {prediction.shape}")  # 打印预测结果的维度

            # 计算损失
            # 1. 交叉重构损失
            loss_cross = criterion_cross(prediction, blendshape_gt)
            # 2. 自我重构损失
            loss_self = criterion_self(prediction, blendshape_gt)
            # 3. 速度损失（时间序列平滑）
            velocity_gt = blendshape_gt[:, 1:] - blendshape_gt[:, :-1]
            velocity_pred = prediction[:, 1:] - prediction[:, :-1]
            loss_velocity = criterion_velocity(velocity_pred, velocity_gt)

            # 总损失
            total_loss_batch = (
                    args.lambda_cross * loss_cross +
                    args.lambda_self * loss_self +
                    args.lambda_velocity * loss_velocity
            )

            # 反向传播和优化
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), args.output_model_path)



# 运行训练
if __name__ == "__main__":
    class Args:
        train_data_path = "./training/data.csv"  # 包含音频路径和 blendshape 路径的映射文件
        output_model_path = "./training/model.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        lr = 1e-4
        batch_size = 8
        epochs = 20
        max_seq_len = 5000  # 设置 max_seq_len 参数
        feature_dim = 832
        bs_dim = 52
        lambda_cross = 1.0
        lambda_self = 1.0
        lambda_velocity = 0.5
        period = 30  # 时间偏置周期


    args = Args()
    train(args)
