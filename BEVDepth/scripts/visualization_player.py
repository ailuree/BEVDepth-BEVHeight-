import cv2
import glob
import os
import numpy as np
from tqdm import tqdm

class VideoGenerator:
    def __init__(self, input_dir="vis_results", output_path="visualization.mp4"):
        self.input_dir = input_dir
        self.output_path = output_path
        self.image_files = []
        
    def load_images(self):
        """加载所有PNG图片"""
        print("正在加载图片...")
        self.image_files = sorted(glob.glob(os.path.join(self.input_dir, "*.png")))
        if not self.image_files:
            raise Exception(f"在 {self.input_dir} 目录下没有找到PNG图片")
        print(f"找到 {len(self.image_files)} 张图片")
        
    def generate_video(self, fps=2):
        """生成视频文件"""
        self.load_images()
        
        # 读取第一张图片来获取尺寸
        first_image = cv2.imread(self.image_files[0])
        height, width = first_image.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        print("正在生成视频...")
        for img_path in tqdm(self.image_files):
            frame = cv2.imread(img_path)
            out.write(frame)
            
        out.release()
        print(f"视频已保存到: {self.output_path}")
        print(f"视频时长: {len(self.image_files)/fps:.1f} 秒")
        print(f"帧率: {fps} FPS")

def main():
    # 创建输出目录
    os.makedirs("video_output", exist_ok=True)
    
    generator = VideoGenerator(
        input_dir="vis_results",
        output_path="video_output/visualization.mp4"
    )
    
    # 可以调整FPS来控制播放速度
    generator.generate_video(fps=2)  # 每秒2帧
    
if __name__ == '__main__':
    main() 