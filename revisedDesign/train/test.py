import librosa
import numpy as np
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torch.optim as optim
from tqdm import tqdm
from model import EmoTalk
from types import SimpleNamespace

wav_path1 = "audio\\angry1.wav"
wav_path2 = "audio\\angry2.wav"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blendshape_path1 = "result\\angry1.npy"
blendshape_path2 = "result\\angry2.npy"

# file_name = wav_path.split('/')[-1].split('.')[0]
speech_array1, sampling_rate = librosa.load(os.path.join(wav_path1), sr=16000)
speech_array2, sampling_rate = librosa.load(os.path.join(wav_path2), sr=16000)

audio1 = torch.FloatTensor(speech_array1).unsqueeze(0).to(device)
audio2 = torch.FloatTensor(speech_array2).unsqueeze(0).to(device)

blendshape1 = torch.tensor(np.load(blendshape_path1),
                           device=device,
                           dtype=torch.float32)
blendshape2 = torch.tensor(np.load(blendshape_path2),
                           device=device,
                           dtype=torch.float32)
# print(audio1.size())
audio1 = audio1.unsqueeze(0)
# print(audio1.size())

audio2 = audio2.unsqueeze(0)
blendshape1 = blendshape1.unsqueeze(0)
blendshape2 = blendshape2.unsqueeze(0)

# print(audio1.size())
# print(blendshape1.size())

# print(audio2.size())
# print(blendshape2.size())

args = SimpleNamespace(feature_dim=832,
                       bs_dim=52,
                       max_seq_len=5000,
                       period=30,
                       batch_size=1,
                       device=device,
                       lr=1e-4,
                       lambda_cross=1.0,
                       lambda_self=1.0,
                       lambda_velocity=0.5,
                       epochs=20,
                       model_path="./pretrain_model/EmoTalk.pth")
level = torch.tensor([1]).to(device)
person = torch.tensor([0], device=device)
model = EmoTalk(args).to(args.device)
model.load_state_dict(torch.load(args.model_path,
                                 map_location=torch.device(args.device)),
                      strict=False)
data = SimpleNamespace(target11=blendshape1,
                       target12=blendshape2,
                       input12=audio1,
                       input21=audio2,
                       level=level,
                       person=person)
data = {**vars(data)}
with torch.no_grad():
    bs_output11, bs_output12, label1 = model.forward(data)
print(blendshape1[0, 1, :])
print(bs_output11[0, 1, :])
# print(bs_output12.size())
# print(label1.size())
# print(label1)
