import os
import subprocess

VIDEO_DIR = "videos"

def compress_video(input_path: str, output_path: str):
    print(f"压缩中: {input_path} → {output_path}")
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "1M",
        "-vf", "scale=-2:480",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ], check=True)
    print(f"完成压缩: {output_path}")

def batch_compress():
    for filename in os.listdir(VIDEO_DIR):
        if not filename.lower().endswith(".mp4"):
            continue
        if filename.endswith("_compressed.mp4"):
            continue
        input_path = os.path.join(VIDEO_DIR, filename)
        output_name = filename.replace(".mp4", "_compressed.mp4")
        output_path = os.path.join(VIDEO_DIR, output_name)

        if os.path.exists(output_path):
            print(f"已存在压缩版本: {output_path}")
            continue

        compress_video(input_path, output_path)

if __name__ == "__main__":
    batch_compress()
