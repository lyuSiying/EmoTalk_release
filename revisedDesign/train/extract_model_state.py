import torch
from pathlib import Path


def extract_model_state(checkpoint_path, output_path):

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # 从checkpoint中提取模型的状态
    model_state = checkpoint['model_state_dict']

    # 仅将模型状态保存到输出路径
    torch.save(model_state, output_path)
    print(f"Model state dict saved to {output_path}")


if __name__ == "__main__":
    # 在训练后，将最终checkpoint转换为仅包含模型参数的 pth 文件
    checkpoint_path = 'pretrain_model/emotalk_model_epoch80_*.pth'  # 最终checkpoint文件名
    output_path = 'pretrain_model/emotalk_model_only_params.pth'  # 模型参数输出路径

    # 通过模式匹配查找正确的checkpoint文件
    checkpoint_files = list(Path('.').glob(checkpoint_path))
    if checkpoint_files:
        checkpoint_file = checkpoint_files[0]
        extract_model_state(checkpoint_file, output_path)  # 提取模型状态并保存
    else:
        print("No checkpoint found for conversion.")
