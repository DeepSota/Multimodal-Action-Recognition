import cv2
import os
import numpy as np

#图片文件夹路径
image_folder_path = '/data2/liuqinglong/multi-modal-tsm/plt_videos/plt/rgb/1'
#最后输出的视频文件路径
videos_folder_path = '/data2/liuqinglong/multi-modal-tsm/plt_videos/videos'
output_video_path = os.path.join(videos_folder_path, 'RGBdelta.mp4')
os.makedirs(videos_folder_path, exist_ok=True)
#获取所有图片文件路径
images = [os.path.join(image_folder_path, img) for img in os.listdir(image_folder_path) if img.endswith(".jpg")]
images.sort()  
#获取合成视频的分辨率
frame = cv2.imread(os.path.join(image_folder_path, images[0]))
height, width, layers = frame.shape
#定义视频编码器和视频对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

# 将图片添加到视频
for image in images:
    video.write(cv2.imread(image))

# 释放视频对象
video.release()
print(f"Video saved at {output_video_path}")
