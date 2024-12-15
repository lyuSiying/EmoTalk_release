import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.optim as optim
from model import EmoTalk
from types import SimpleNamespace
import random
import os
from datetime import datetime
from pathlib import Path


# EmoTalk 数据集类
class EmoTalkDataset(Dataset):

    def __init__(self, audio_files, blendshape_files, device):
        # 初始化音频文件和blendshape文件
        self.audio_files = audio_files
        self.blendshape_files = blendshape_files
        self.device = device
        self.file_pairs = list(zip(audio_files,
                                   blendshape_files))  # 将音频和blendshape文件配对
        random.shuffle(self.file_pairs)  # 初始时打乱文件对
        self.used_pairs = set()  # 用于追踪在一个轮次中已使用的文件对

    def __len__(self):
        # 返回样本的数量，每个样本由两个音频文件组成
        return len(self.file_pairs) // 2

    def __getitem__(self, index):
        # 当轮次结束时，清空已使用的文件对
        if len(self.used_pairs) >= len(self.file_pairs) - 1:
            self.used_pairs.clear()

        while True:
            # 随机选择两个不同的文件对
            pair_indices = random.sample(range(len(self.file_pairs)), 2)
            pair_1, pair_2 = [self.file_pairs[i] for i in pair_indices]

            # 确保选择的文件对是唯一的，并且在本轮中未被使用
            if (pair_1, pair_2) not in self.used_pairs and (
                    pair_2, pair_1) not in self.used_pairs:
                self.used_pairs.add((pair_1, pair_2))  # 添加到已使用的文件对集合
                break

        # 加载第一个音频文件和对应的blendshape
        wav_path_1, blendshape_path_1 = pair_1
        speech_array_1, sampling_rate = librosa.load(wav_path_1, sr=16000)
        audio_1 = torch.FloatTensor(speech_array_1).unsqueeze(0).to(
            self.device)

        blendshape_1 = np.load(blendshape_path_1)
        blendshape_tensor_1 = torch.tensor(blendshape_1,
                                           device=self.device,
                                           dtype=torch.float32)

        # 加载第二个音频文件和对应的blendshape
        wav_path_2, blendshape_path_2 = pair_2
        speech_array_2, sampling_rate = librosa.load(wav_path_2, sr=16000)
        audio_2 = torch.FloatTensor(speech_array_2).unsqueeze(0).to(
            self.device)

        blendshape_2 = np.load(blendshape_path_2)
        blendshape_tensor_2 = torch.tensor(blendshape_2,
                                           device=self.device,
                                           dtype=torch.float32)

        return {
            "input12": audio_1,  # 输入12
            "input21": audio_2,  # 输入21
            "target11": blendshape_tensor_1,  # 目标11
            "target12": blendshape_tensor_2,  # 目标12
        }


# EmoTalk 损失类
class EmotalkLoss():

    def __init__(self, args):
        # 初始化损失权重参数
        self.lambda_cross = args.lambda_cross
        self.lambda_self = args.lambda_self
        self.lambda_velocity = args.lambda_velocity
        self.lambda_cls = args.lambda_cls
        self.mse_loss = nn.MSELoss()  # 均方误差损失
        self.cross_entropy_loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def Loss(self, outputs, targets, labels):
        # 提取输出和目标张量
        D12 = outputs["output12"]
        D21 = outputs["output21"]
        D11 = outputs["output11"]
        B1 = targets["target1"]
        B2 = targets["target2"]

        # 交叉重构损失
        L_cross = self.mse_loss(D12, B1) + self.mse_loss(D21, B2)

        # 自我重构损失
        L_self = self.mse_loss(D11, B1)

        # 速度损失
        velocity_gt = B1[:, 1:] - B1[:, :-1]  # 真实速度
        velocity_pred = D12[:, 1:] - D12[:, :-1]  # 预测速度
        L_velocity = self.mse_loss(velocity_gt, velocity_pred)

        # 分类损失
        # L_cls = self.cross_entropy_loss(D12, labels)

        # 合计总损失
        loss = (L_cross * self.lambda_cross + L_self * self.lambda_self +
                L_velocity * self.lambda_velocity)  # + L_cls * self.lambda_cls

        return loss


# 前向传播的函数
def forward_pass(model, inputs12, inputs21, targets11, targets12, device):
    # 执行模型的前向传播

    #用于交叉重构损失和速度损失
    bs_output11, bs_output12, label1 = model({
        "input12":
        inputs12,
        "input21":
        inputs21,
        "target11":
        targets11,
        "target12":
        targets12,
        "level":
        torch.tensor([1]).to(device),
        "person":
        torch.tensor([0], device=device),
    })

    #用于交叉重构损失
    bs_output21, bs_output22, label2 = model({
        "input12":
        inputs21,
        "input21":
        inputs12,
        "target11":
        targets12,
        "target12":
        targets11,
        "level":
        torch.tensor([1]).to(device),
        "person":
        torch.tensor([0], device=device),
    })

    #用于速度损失
    bs_output_self1, _, label3 = model({
        "input12":
        inputs12,
        "input21":
        inputs12,
        "target11":
        targets11,
        "target12":
        targets11,
        "level":
        torch.tensor([1]).to(device),
        "person":
        torch.tensor([0], device=device),
    })

    # 返回输出和标签
    return {
        "output12": bs_output11,
        "output21": bs_output21,
        "output11": bs_output_self1
    }, [label1, label2, label3]


# 加载checkpoint的函数
def load_checkpoint(model_path, model, optimizer, device):
    start_epoch = 0
    best_loss = float('inf')

    # 定义checkpoint文件的目录和基础名称
    model_dir = Path(model_path).parent
    model_name = Path(model_path).stem  # 获取文件名不带扩展名

    # 构建匹配checkpoint文件的模式
    checkpoint_pattern = f"{model_name}_epoch*.pth"

    # 尝试找到最新的checkpoint文件
    checkpoint_files = list(model_dir.glob(checkpoint_pattern))

    if checkpoint_files:
        # 直接加载检测到的checkpoint文件
        checkpoint_file = checkpoint_files[0]

        print(f"Loading model parameters from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])  # 加载模型状态
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])  # 加载优化器状态
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        print(
            f"Resuming training from epoch {start_epoch} with last loss: {best_loss:.4f}"
        )
    else:
        print("No existing model found. Initializing new model.")

    return start_epoch, best_loss, model, optimizer


# 保存checkpoint的函数
def save_checkpoint(epoch, model, optimizer, loss, model_path):
    # 生成时间戳并将epoch数包含在checkpoint中
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_model_path = f"{model_path}_epoch{epoch}_{timestamp}.pth"

    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)  # 确保保存目录存在

    try:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, new_model_path)  # 保存checkpoint
        print(f'Checkpoint saved to {new_model_path} at {datetime.now()}')

        # 如果存在前一轮的检checkpoint，则删除它
        if epoch > 1:  # 确保在第二轮之前不会尝试删除
            prev_epoch = epoch - 1
            prev_model_pattern = f"{model_path}_epoch{prev_epoch}_*.pth"
            prev_model_files = list(Path('.').glob(prev_model_pattern))

            for prev_model_file in prev_model_files:
                os.remove(prev_model_file)  # 删除旧的checkpoint
                print(f"Removed old checkpoint at {prev_model_file}")
    except Exception as e:
        print(f"Failed to save checkpoint: {e}")


# 训练模型的函数
def train_model(args):
    model = EmoTalk(args).to(args.device)  # 初始化模型并将其放置到设备上
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # 使用 Adam 优化器

    # 如果存在checkpoint，则加载
    start_epoch, best_loss, model, optimizer = load_checkpoint(
        args.model_path, model, optimizer, args.device)

    dataset = EmoTalkDataset(audio_files=args.audio_files,
                             blendshape_files=args.blendshape_files,
                             device=args.device)  # 初始化数据集
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False)  # 使用 DataLoader 进行批处理
    num_epochs = args.epochs
    Loss = EmotalkLoss(args)  # 初始化损失对象
    model.train()  # 设置模型为训练模式

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        total_batches = 0

        # 在每个轮次开始时重置已使用的文件对
        dataset.used_pairs.clear()

        for batch in dataloader:
            # 获取输入、目标值
            inputs12 = batch["input12"]
            inputs21 = batch["input21"]
            targets11 = batch["target11"]
            targets12 = batch["target12"]

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs, labels = forward_pass(model, inputs12, inputs21,
                                           targets11, targets12, args.device)

            # 计算损失
            loss = Loss.Loss(outputs, {
                "target1": targets11,
                "target2": targets12
            }, labels[0])

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches if total_batches > 0 else float(
            'inf')  # 计算平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 每个轮次结束后保存checkpoint
        save_checkpoint(epoch + 1, model, optimizer, avg_loss, args.model_path)


# 主程序
def main():

    audio_files = list(Path('audio').glob('*.wav'))  # 获取所有音频文件
    blendshape_files = list(Path('result').glob('*.npy'))  # 获取所有blendshape文件

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # 选择设备

    # 设置训练参数
    args = SimpleNamespace(
        audio_files=audio_files,
        blendshape_files=blendshape_files,
        feature_dim=832,
        bs_dim=52,
        max_seq_len=5000,
        period=30,
        batch_size=1,  # 设为1以支持可变长度序列
        device=device,
        lr=1e-4,  # 学习率
        lambda_cross=1.0,  # 交叉重构损失的权重
        lambda_self=1.0,  # 自我重构损失的权重
        lambda_velocity=0.5,  # 速度损失的权重
        lambda_cls=1.0,  # 分类损失的权重
        epochs=80,  # 训练的总轮次
        model_path='pretrain_model/emotalk_model')  # 模型保存路径

    train_model(args)  # 开始训练模型


if __name__ == "__main__":
    main()
