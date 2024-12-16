import os
import subprocess
from pathlib import Path


def convert_mp4_to_wav(mp4_file, output_dir):
    """
    使用 ffmpeg 将单个 MP4 文件转换为 WAV 文件。

    参数:
    mp4_file (Path): 输入的 MP4 文件路径。
    output_dir (Path): 输出的 WAV 文件保存目录。
    """
    if not mp4_file.is_file() or mp4_file.suffix.lower() != '.mp4':
        print(f"Skipping non-MP4 file: {mp4_file}")
        return

    output_wav_path = output_dir / (mp4_file.stem + ".wav")

    command = [
        'ffmpeg', '-i',
        str(mp4_file), '-q:a', '0', '-map', 'a', '-ac', '1', '-ar', '16000',
        str(output_wav_path)
    ]

    try:
        subprocess.run(command,
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
        print(f"Converted: {mp4_file} -> {output_wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {mp4_file}: {e.stderr.decode()}")


def batch_convert_mp4_to_wav(input_dir, output_dir):
    """
    批量将输入目录中的所有 MP4 文件转换为 WAV 文件，并保存到输出目录中。

    参数:
    input_dir (str or Path): 包含 MP4 文件的输入目录。
    output_dir (str or Path): 输出 WAV 文件的保存目录。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"The directory {input_dir} does not exist.")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    mp4_files = list(input_dir.glob('*.mp4'))

    if not mp4_files:
        print(f"No MP4 files found in the directory: {input_dir}")
        return

    for mp4_file in mp4_files:
        convert_mp4_to_wav(mp4_file, output_dir)


def main():
    input_directory = "data/raw/videos"
    output_directory = "test_audio"

    batch_convert_mp4_to_wav(input_directory, output_directory)


if __name__ == "__main__":
    main()
