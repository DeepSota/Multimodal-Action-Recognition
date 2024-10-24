import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np


root_dir = r"/data2/wsp/multi-modal-tsm/plt_videos/original/ir/30"
save_dir = r"/data2/wsp/multi-modal-tsm/plt_videos/plt/ir/30"
start_num = 1
end_num = 150
image_format = ".jpg"

os.makedirs(save_dir, exist_ok=True)


for i in range(start_num, end_num+1):
    image_filename = f"{i:06d}{image_format}"
    image_path = os.path.join(root_dir,image_filename) 

    try:
        image = Image.open(image_path)
        plt.imshow(image)
        if i % 2 == 0:
            ax = plt.gca()
            #在图片上添加矩形框和标签  RGB #(5,5), 440, 240 # 10, 20   IR# (5,5), 310, 240,  10, 17,
            ax.add_patch(plt.Rectangle( (5,5), 310, 240, color = "green", fill = False, linewidth = 1))
            ax.text(10, 17, f"label: tie hair", bbox = {'facecolor': 'green', 'alpha': 0.5})
        
        plt.axis('off')
        save_filename = image_filename
        save_path = os.path.join(save_dir,save_filename)
        plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
        plt.close()
    
    except IOError:
        print(f"Error in opening or processing the file {image_path}. It may not exist or be corrupted.")

    print('1')






# adv

# root_dir = r"/data2/liuqinglong/multi-modal-tsm/plt_videos/original/rgb/1"
# save_dir = r"/data2/liuqinglong/multi-modal-tsm/plt_videos/plt/rgb/1"
# start_num = 1
# end_num = 150
# image_format = ".jpg"
# os.makedirs(save_dir, exist_ok=True)
# #获取所有扰动的图片
# delta_folder_path = '/data2/liuqinglong/multi-modal-tsm/plt_videos/delta/rgb_32'
# delta_images = [os.path.join(delta_folder_path, img) for img in os.listdir(delta_folder_path) if img.endswith(".jpg")]
# delta_images.sort()


# for i in range(start_num, end_num+1):
#     image_filename = f"{i:06d}{image_format}"
#     image_path = os.path.join(root_dir,image_filename) 

#     try:
#         image = Image.open(image_path)
#         image_npy= np.array(image)

#         delta_image = Image.open(delta_images[i%8]).resize(( 455,256), Image.ANTIALIAS)
#         delta_image_npy= np.array(delta_image)*0.3
#         image_npy =(image_npy+delta_image_npy).clip(0, 255).astype(np.uint8)
#         image = Image.fromarray(image_npy)

#         plt.imshow(image)
#         if i % 2 == 0:
#             ax = plt.gca()
#             #在图片上添加矩形框和标签  RGB #(5,5), 440, 240 # 10, 20   IR# (5,5), 310, 240,  10, 17,
#             ax.add_patch(plt.Rectangle((5,5), 440, 240, color = "red", fill = False, linewidth = 1))
#             ax.text(10, 20, f"label: tie shoes", bbox = {'facecolor': 'red', 'alpha': 0.5})
        
#         plt.axis('off')
#         save_filename = image_filename
#         save_path = os.path.join(save_dir,save_filename)
#         plt.savefig(save_path, bbox_inches = 'tight', pad_inches = 0)
#         plt.close()
    
#     except IOError:
#         print(f"Error in opening or processing the file {image_path}. It may not exist or be corrupted.")

#     print('1')















