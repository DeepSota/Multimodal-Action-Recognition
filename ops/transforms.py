import itertools
import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert(img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):

        # 可视化

        # ii = 0
        # for ret_img in img_group:
        #     ret_img.save('save_img/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1

        img_group = [self.worker(img) for img in img_group]

        # for ret_img in img_group:
        #     ret_img.save('save_img2/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1

        return   img_group


class GroupWindowCrop(object):
    def __init__(self, final_size, window_size, num_parts ):
        self.worker = torchvision.transforms.CenterCrop(final_size)
        self.final_size = final_size
        self.window_size = window_size
        self.num_parts = num_parts

    def split_image(self, img,):
        """
        将图片使用滑动窗口进行分割。
        
        :param img: PIL Image对象
        :param window_size: 滑动窗口的尺寸 (width, height)
        :param num_parts: 分割的份数
        :return: 分割后的小块列表
        """
        num_parts = self.num_parts 
        width, height = img.size
     
        
        # y 为 2 
        # h_part = 2
        # w_part= num_parts//2

        # step_x=[] 
        # dengfen_juli =width - self.window_size
        # dengfen_x = int(dengfen_juli//(w_part-1))
        # zhongxin_x_step = dengfen_x
        # zhongxin_x = self.window_size//2
        # if w_part>=2:
        #     for i in range(w_part-1):                       
        #                 step_x.append(zhongxin_x)
        #                 zhongxin_x +=zhongxin_x_step   
        # step_x.append(width-self.window_size//2) 

        # step_y = [self.window_size//2, height-self.window_size//2]
        # crops = []
        # for i in range(h_part):
        #     for j in range(w_part):
        #         left = step_x[j] - self.window_size//2
        #         upper = step_y[i] - self.window_size//2
        #         right = step_x[j] + self.window_size//2
        #         lower = step_y[i] + self.window_size//2
                
        #         crop = img.crop((left, upper, right, lower))
        #         crops.append(crop)

        h_part = 1
        w_part= num_parts

        step_x=[] 
        dengfen_juli =width - self.window_size
        dengfen_x = int(dengfen_juli//(w_part-1))
        zhongxin_x_step = dengfen_x
        zhongxin_x = self.window_size//2
        if w_part>=2:
            for i in range(w_part-1):                       
                        step_x.append(zhongxin_x)
                        zhongxin_x +=zhongxin_x_step   
        step_x.append(width-self.window_size//2) 

        step_y = [height//2]
        crops = []
        for i in range(h_part):
            for j in range(w_part):
                left = step_x[j] - self.window_size//2
                upper = step_y[i] - self.window_size//2
                right = step_x[j] + self.window_size//2
                lower = step_y[i] + self.window_size//2
                
                crop = img.crop((left, upper, right, lower))
                crops.append(crop)

      


        # 可视化      
        #   
        # ii = 0
        # for ret_img in crops:
        #     ret_img.save('save_img/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1

        return crops

    def __call__(self, img_group):
        
        # 可视化

        # ii = 0
        # for ret_img in img_group:
        #     ret_img.save('save_img/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1
        crops_list = []
        for img in img_group:
            crops = self.split_image(img)
            crops_list.append(crops)

        group_centercrop = [self.worker(img) for img in img_group]

        concatenated_lists = [col for col in zip(*crops_list)]
        concatenated_lists =  [item for sublist in concatenated_lists for item in sublist]
        concatenated_lists= concatenated_lists+group_centercrop

        # 可视化

        # ii = 0
        # for ret_img in concatenated_lists:
        #     ret_img.save('save_img2/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1

        return   concatenated_lists
    
class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[1]//len(self.mean))
        rep_std = self.std * (tensor.size()[1]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)


        # # # 可视化 
        # for t, m, s in zip(tensor, rep_mean, rep_std):
        #     t.mul_(s).add_(m)

        # img =tensor.view(-1,9,224,224)
        # x_rgbs, x_irs, x_depths = img[:,:3,:,:], img[:,3:6,:,:], img[:,6:9,:,:]
        # # 保存图像
        # i = 0
        # ii= 0
        # iii =0 

        # for x_rgb in x_rgbs:
        #     x_rgb = x_rgb*255
        #     x_rgb  = Image.fromarray(x_rgb.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        #     x_rgb.save('save_imgs/rgb/{}.jpg'.format(i), 'JPEG')
        #     i+=1

        # for x_ir in x_irs:
        #     x_ir = x_ir*255
        #     x_ir  = Image.fromarray(x_ir.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        #     x_ir.save('save_imgs/ir/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1

        # for x_depth in x_depths:
        #     x_depth = x_depth*255
        #     x_depth = Image.fromarray( x_depth.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
        #     x_depth.save('save_imgs/depth/{}.jpg'.format(iii), 'JPEG')
        #     iii+=1

        return tensor
    
class Reverse_GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        rep_mean = self.mean * (tensor.size()[0]//len(self.mean))
        rep_std = self.std * (tensor.size()[0]//len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.mul_(s).add_(m) 

        return tensor

class GroupScale(object):
    """ Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            if self.flip:
                oversample_group.extend(flip_group)
        return oversample_group

class Padding(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        padded_images = []
        ii = 0
        for img in img_group:
            # 获取图片的宽度和高度
            width, height = img.size
            
            # 计算短边和长边
            if width < height:
                long_edge = height
                short_edge = width
                # 计算需要padding的尺寸
                pad_width = (long_edge - short_edge) // 2
                pad_height = pad_width
                paste_x = pad_width
                paste_y = 0
            else:
                long_edge = width
                short_edge = height
                pad_width = (long_edge - short_edge) // 2
                pad_height = pad_width
                paste_x = 0
                paste_y = pad_height
            

            
            # 创建一个新的Image对象，用于padding
            padding_color=(0, 0, 0)
            padded_img = Image.new('RGB', (long_edge, long_edge), padding_color)
            # 计算粘贴的起始位置
            padded_img.paste(img, (paste_x, paste_y))

            # padded_img.save('save_img/{}.jpg'.format(ii), 'JPEG')
            # ii+=1
            padded_images.append(padded_img)

        
        return padded_images


class GroupFullResSample(object):
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                if self.flip:
                    flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                    if img.mode == 'L' and i % 2 == 0:
                        flip_group.append(ImageOps.invert(flip_crop))
                    else:
                        flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        # ret_img_group = []
        # for img in img_group:
        #     im_size = img.size
        #     crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        #     crop_img = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) 
        #     ret_img =  crop_img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        #     ret_img_group.append(ret_img)                
        

        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                        for img in crop_img_group]
        
        # 可视化
        # ii = 0
        # for ret_img in ret_img_group:
        #     ret_img.save('save_img/{}.jpg'.format(ii), 'JPEG')
        #     ii+=1
        
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)

  




class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()

            # 可视化 
            # img =img.reshape(-1,9,224,224)
            # x_rgbs, x_irs, x_depths = img[:,:3,:,:], img[:,3:6,:,:], img[:,6:9,:,:]
            # i = 0
            # ii=0
            # iii=0
            # for x_rgb in x_rgbs:
                
            #     x_rgb  = Image.fromarray(x_rgb.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
            #     x_rgb.save('save_imgs/rgb/{}.jpg'.format(i), 'JPEG')
            #     i+=1

            # for x_ir in x_irs:
              
            #     x_ir  = Image.fromarray(x_ir.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
            #     x_ir.save('save_imgs/ir/{}.jpg'.format(ii), 'JPEG')
            #     ii+=1

            # for x_depth in x_depths:
             
            #     x_depth = Image.fromarray( x_depth.permute(1, 2, 0).detach().cpu().numpy().astype('uint8'))
            #     x_depth.save('save_imgs/depth/{}.jpg'.format(iii), 'JPEG')
            #     iii+=1

        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):

    def __call__(self, data):
        return data


if __name__ == "__main__":
    trans = torchvision.transforms.Compose([
        GroupScale(256),
        GroupRandomCrop(224),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225]
        )]
    )

    im = Image.open('../tensorflow-model-zoo.torch/lena_299.png')

    color_group = [im] * 3
    rst = trans(color_group)

    gray_group = [im.convert('L')] * 9
    gray_rst = trans(gray_group)

    trans2 = torchvision.transforms.Compose([
        GroupRandomSizedCrop(256),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(
            mean=[.485, .456, .406],
            std=[.229, .224, .225])
    ])
    print(trans2(color_group))