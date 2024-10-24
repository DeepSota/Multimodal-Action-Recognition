import torch
from PIL import Image

def Reverse_GroupNormalize(tensor):
    mean =  [0.485, 0.456, 0.406]
    std =   [0.229, 0.224, 0.225]
    rep_mean = mean * (tensor.size()[1]//len(mean))
    rep_std = std * (tensor.size()[1]//len(std))

    # TODO: make efficient
   
    for t, m, s in zip(tensor, rep_mean, rep_std):
            t.mul_(s).add_(m)
            # t = (t-min(t))/(max(t)-min(t))
            # t = t.mul(s).add(m)
            t.clamp(0, 1)
    return tensor
    

def save_imgs(img_tensors):
    img_tensors = img_tensors.reshape(-1, 9, 224, 224)
    x = Reverse_GroupNormalize(img_tensors)
    # x = img_tensors.view(-1, 9, 224, 224)
    x_rgbs, x_irs, x_depths = x[:,:3,:,:], x[:,3:6,:,:], x[:,6:9,:,:]

    # 保存图像
    i = 0
    ii= 0
    iii =0 

    for x_rgb in x_rgbs:
        x_rgb = x_rgb*255
        x_rgb  = Image.fromarray(x_rgb.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        x_rgb.save('save_imgs/rgb/{}.jpg'.format(i), 'JPEG')
        i+=1

    for x_ir in x_irs:
        x_ir = x_ir*255
        x_ir  = Image.fromarray(x_ir.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        x_ir.save('save_imgs/ir/{}.jpg'.format(ii), 'JPEG')
        ii+=1

    for x_depth in x_depths:
        x_depth = x_depth*255
        x_depth = Image.fromarray( x_depth.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        x_depth.save('save_imgs/depth/{}.jpg'.format(iii), 'JPEG')
        iii+=1

    return 0 