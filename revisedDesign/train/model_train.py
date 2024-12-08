import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
from model import EmoTalk
from types import SimpleNamespace
import random
from collections import defaultdict


class EmoTalkDataset(Dataset):

    def __init__(self, audio_files, blendshape_files, device):
        self.audio_files = audio_files
        self.blendshape_files = blendshape_files
        self.device = device
        self.file_pairs = list(zip(audio_files, blendshape_files))
        random.shuffle(self.file_pairs)  # Shuffle the pairs initially
        self.used_pairs = set()  # Track used pairs within an epoch

    def __len__(self):
        return len(
            self.file_pairs) // 2  # Each sample contains two audio files

    def __getitem__(self, index):
        if len(self.used_pairs) >= len(self.file_pairs) - 1:
            self.used_pairs.clear(
            )  # Reset used pairs at the end of each epoch

        while True:
            # Randomly select two different file pairs
            pair_indices = random.sample(range(len(self.file_pairs)), 2)
            pair_1, pair_2 = [self.file_pairs[i] for i in pair_indices]

            # Ensure the selected pairs are unique and not already used this epoch
            if (pair_1, pair_2) not in self.used_pairs and (
                    pair_2, pair_1) not in self.used_pairs:
                self.used_pairs.add((pair_1, pair_2))
                break

        # Load first audio file and corresponding blendshape
        wav_path_1, blendshape_path_1 = pair_1
        speech_array_1, sampling_rate = librosa.load(wav_path_1, sr=16000)
        audio_1 = torch.FloatTensor(speech_array_1).unsqueeze(0).to(
            self.device)

        blendshape_1 = np.load(blendshape_path_1)
        blendshape_tensor_1 = torch.tensor(blendshape_1,
                                           device=self.device,
                                           dtype=torch.float32)

        # Load second audio file and corresponding blendshape
        wav_path_2, blendshape_path_2 = pair_2
        speech_array_2, sampling_rate = librosa.load(wav_path_2, sr=16000)
        audio_2 = torch.FloatTensor(speech_array_2).unsqueeze(0).to(
            self.device)

        blendshape_2 = np.load(blendshape_path_2)
        blendshape_tensor_2 = torch.tensor(blendshape_2,
                                           device=self.device,
                                           dtype=torch.float32)

        return {
            "input12": audio_1,
            "input21": audio_2,
            "target11": blendshape_tensor_1,
            "target12": blendshape_tensor_2,
        }


class EmotalkLoss():

    def __init__(self, args):
        self.lambda_cross = args.lambda_cross
        self.lambda_self = args.lambda_self
        self.lambda_velocity = args.lambda_velocity
        self.lambda_cls = args.lambda_cls
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def Loss(self, outputs, targets, labels):
        D12 = outputs["output12"]
        D21 = outputs["output21"]
        D11 = outputs["output11"]
        B1 = targets["target1"]
        B2 = targets["target2"]

        # Cross-reconstruction loss
        L_cross = self.mse_loss(D12, B1) + self.mse_loss(D21, B2)

        # Self-reconstruction loss
        L_self = self.mse_loss(D11, B1)

        # Velocity loss
        velocity_gt = D12[:, 1:] - B1[:, :-1]
        velocity_pred = D12[:, 1:] - B1[:, :-1]
        L_velocity = self.mse_loss(velocity_gt, velocity_pred)

        # Classification loss
        # L_cls = self.cross_entropy_loss(D12, labels)

        loss = L_cross * self.lambda_cross + L_self * self.lambda_self + L_velocity * self.lambda_velocity  #+ L_cls * self.lambda_cls

        return loss


def train_model(args):
    model = EmoTalk(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_cross = nn.MSELoss()  # 用于交叉重构损失

    dataset = EmoTalkDataset(audio_files=args.audio_files,
                             blendshape_files=args.blendshape_files,
                             device=args.device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    num_epochs = args.epochs
    Loss = EmotalkLoss(args)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # Reset used pairs at the beginning of each epoch
        dataset.used_pairs.clear()

        for batch in progress_bar:
            # 获取输入、目标值和帧数
            inputs12 = batch["input12"]
            inputs21 = batch["input21"]
            targets11 = batch["target11"]
            targets12 = batch["target12"]

            # 零梯度
            optimizer.zero_grad()

            # 前向传播
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
                torch.tensor([1]).to(args.device),  # 这里需要根据实际情况调整
                "person":
                torch.tensor([0], device=args.device),  # 这里需要根据实际情况调整
            })

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
                torch.tensor([1]).to(args.device),  # 这里需要根据实际情况调整
                "person":
                torch.tensor([0], device=args.device),  # 这里需要根据实际情况调整
            })

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
                torch.tensor([1]).to(args.device),  # 这里需要根据实际情况调整
                "person":
                torch.tensor([0], device=args.device),  # 这里需要根据实际情况调整
            })
            outputs = {
                "output12": bs_output11,
                "output21": bs_output21,
                "output11": bs_output_self1
            }

            targets = {"target1": targets11, "target2": targets12}

            # 计算损失
            loss = Loss.Loss(outputs, targets, label1)

            # 反向传播
            loss.backward()

            # 参数更新
            optimizer.step()

            # 统计损失
            running_loss += loss.item()
            avg_loss = running_loss / (progress_bar.n + 1)  # 使用当前进度条的位置计算平均损失
            progress_bar.set_postfix({'loss': avg_loss})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


def main():
    # Assuming that the audio and blendshape files are paired by their order in the directory
    audio_files = list(Path('audio').glob('*.wav'))
    blendshape_files = list(Path('result').glob('*.npy'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = SimpleNamespace(
        audio_files=audio_files,
        blendshape_files=blendshape_files,
        feature_dim=832,
        bs_dim=52,
        max_seq_len=5000,
        period=30,
        batch_size=1,  # Set to 1 for variable length sequences
        device=device,
        lr=1e-4,
        lambda_cross=1.0,
        lambda_self=1.0,
        lambda_velocity=0.5,
        lambda_cls=1.0,
        epochs=2)

    train_model(args)


if __name__ == "__main__":
    main()
