import argparse
import csv
import torch
import torchvision
from torch.nn import functional as F
from tqdm import tqdm 
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config_for_pred as dataset_config

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('--dataset', default='mmvpr', type=str)
parser.add_argument('--weights', type=str, default='/data2/wsp/multi-modal-tsm/checkpoint/TSM_mmvpr_RTD_mobilenetv2_shift8_blockres_avg_segment8_e60/ckpt_59.pth.tar')
parser.add_argument('--test_segments', type=int, default=8)
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--workers', default=8, type=int, metavar='N')
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default='submission.csv')
parser.add_argument('--softmax', default=False, action="store_true")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--test_file', type=str, default='ICPR_MMVPR_Track3/test_set/test.txt')
args = parser.parse_args()


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None

args.modality = 'RTD'
# if 'RGB' in args.weights:
#     args.modality = 'RGB'd
# elif 'Flow' in args.weights:
#     args.modality = 'Flow'
# else:
#     args.modality = 'RTD'
num_class, args.train_list, args.val_list, args.root_path, args.root_data_depth, args.root_data_ir, prefix, prefix_ir, prefix_depth = dataset_config.return_dataset(args.dataset, args.modality)

# -----------------------------------------------------------------------------------------
weights_lits=[
               'checkpoint/TSM_mmvpr_RTD_resnet50_shift8_blockres_avg_segment8_e300.95depth0.2/ckpt_best.pth.tar', # 0.95
               'checkpoint/TSM_mmvpr_RTD_resnet101_shift8_blockres_avg_segment8_e30/ckpt_best.pth.tar'
               # 'checkpoint/TSM_mmvpr_RTD_resnet50_shift8_blockres_avg_segment8_e50_nl/ckpt_swa.pth.tar'
             
              ]
net_list= []

# net1 加載
args.weights= weights_lits[0]
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(args.weights)
this_arch = args.weights.split('TSM_')[1].split('_')[2]
net = TSN(num_class, args.test_segments if is_shift else 1, args.modality,
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in args.weights)
checkpoint = torch.load(args.weights)
checkpoint = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias'}
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net.load_state_dict(base_dict)
net.eval()
net_list.append(net)
# -----------------------------------------------------------------------------------------
# net2 加載

args.weights= weights_lits[1]
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(args.weights)
this_arch = args.weights.split('TSM_')[1].split('_')[2]
net2 = TSN(num_class, args.test_segments if is_shift else 1, args.modality,
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in args.weights)
checkpoint = torch.load(args.weights)
checkpoint = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias'}
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)

net2.load_state_dict(base_dict)
net2.eval()
net_list.append(net2)

# -----------------------------------------------------------------------------------------

input_size = net.scale_size if args.full_res else net.input_size
cropping = torchvision.transforms.Compose([
    GroupScale(net.scale_size),
    # GroupCenterCrop(input_size),
    GroupWindowCrop(input_size, 224, 2)
])

if args.modality != 'RGBDiff':
    normalize = GroupNormalize(net.input_mean, net.input_std)
else:
    normalize = IdentityTransform()

if args.modality in ['RGB', 'RTD']:
    data_length = 1
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5


data_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.root_data_ir, args.root_data_depth, list_file=args.test_file if args.test_file is not None else args.val_list, num_segments=args.test_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix, image_tmpl_ir=prefix_ir, image_tmpl_depth=prefix_depth,
               test_mode=True,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                   normalize,
               ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

# net = torch.nn.DataParallel(net.cuda())


output = []

def eval_video(video_data, net_list, this_test_segments):
    with torch.no_grad():
        i, data = video_data
        batch_size = args.batch_size
        num_crop = args.test_crops

        if args.modality == 'RGB':
            sample_length = 3
        elif args.modality == 'RTD':
            sample_length = 9
        elif args.modality == 'Flow':
            sample_length = 10
        else:
            raise ValueError("Unknown modality "+ args.modality)
        data = data[0]

        # data_in = data[0].view(-1, sample_length, data.size(2), data.size(3))
        data_in_list = data[0].view(batch_size, -1, data.size(2), data.size(3))
        # print('data_in', data_in.size())
        if is_shift:
            data_in_list = data_in_list.view( -1, batch_size,  this_test_segments, sample_length, data_in_list.size(2), data_in_list.size(3))


        # 模型融合

        #TTA 翻折
        alpha = 1
        view_modes = data_in_list.size(0)
        fis=[4, None] # 水平翻折
        rsts = 0.0
        for view_i in range(view_modes):
            data_in = data_in_list[view_i]
            if view_i >= view_modes-1:
               alpha = (view_modes-1)*2
            for fi in fis:
                for net in net_list:
                    if fi:
                       data_in=data_in.flip(fi)
                    out = net(data_in)
                    out = F.softmax(out, dim=1)
                    rsts += alpha*out

        rst= rsts/(len(fis)*len(net_list))

        # rst = net_list[0](data_in)

        # rst0 = net_list[0](data_in)
        # rst1 = net_list[1](data_in)
        # rst0 = F.softmax(rst0, dim=1)
        # rst1 = F.softmax(rst1, dim=1)
        # rst =(rst0+rst1)/2
       
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        # if args.softmax:
        #     rst = F.softmax(rst, dim=1)

        return i, rst.data.cpu().numpy().copy()

# for i, data in enumerate(data_loader):
#     rst = eval_video((i+1, data), net_list, args.test_segments)
#     output.append([rst[1]]) 

# pbar = tqdm(data_loader, desc='Evaluating')
# for i, data in enumerate(pbar):
#     rst = eval_video((i+1, data), net_list, args.test_segments)
#     output.append([rst[1]])
#     pbar.set_description(f'Evaluating - Batch size: {args.batch_size}')

print(weights_lits)
print(f'save result path {args.csv_file}')
for i, data in enumerate(tqdm(data_loader, desc='Evaluating')):
    rst = eval_video((i+1, data), net_list, args.test_segments)
    output.append([rst[1]])
    

video_pred =  [np.argsort(x[0][0])[::-1][:5] for x in output]   
vid_id = 0
with open(args.csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Video', 'Prediction'])
    for  pred1,pred2,pred3,pred4,pred5, in video_pred:
        vid_id+=1
        csvwriter.writerow([vid_id, '{} {} {} {} {}'.format(pred1,pred2,pred3,pred4,pred5)])

print(f'Predictions saved to {args.csv_file}')


# video_pred =  [np.argmax(x[0], axis=1)[0] for x in output] # [np.argsort(x[0][0])[::-1][:5] for x in output]  
# vid_id = 0
# with open(args.csv_file, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(['Video', 'Prediction'])
#     for  pred in video_pred:
#         vid_id+=1
#         csvwriter.writerow([vid_id, pred])

# print(f'Predictions saved to {args.csv_file}')
