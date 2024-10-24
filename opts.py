# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--dataset', default='mmvpr', type=str)
parser.add_argument('--modality', default='RTD', type=str, choices=['RGB','Flow','Depth','IR','RTD'])
parser.add_argument('--train_list', type=str, default="")
parser.add_argument('--val_list', type=str, default="")
parser.add_argument('--root_path', type=str, default="")
parser.add_argument('--root_path_depth', type=str, default="")
parser.add_argument('--root_path_ir', type=str, default="")
# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet50")  #mobilenetv2 resnet50 resnet101 resnet18 BNInception
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='avg')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)') 
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")
parser.add_argument('--suffix', type=str, default=None)
parser.add_argument('--pretrain', type=str, default='imagenet')

# ========================= tune_from weiths ==========================

# parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth')
parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth')
# parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment16_e45.pth')
# parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth')
# parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_somethingv2_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth')
# parser.add_argument('--tune_from', type=str, default='/data2/wsp/multi-modal-tsm/weights/TSM_somethingv1_RGB_resnet101_shift8_blockres_avg_segment8_e45.pth')



# ========================= Learning Configs ==========================


# ========================= Learning Configs ==========================
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=6, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_type', default='step', type=str,
                    metavar='LRtype', help='learning rate type')
parser.add_argument('--lr_steps', default=[10, 20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=True, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=50, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', default=[0], type=int, )
parser.add_argument('--flow_prefix', default="", type=str)
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')

# parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--shift', default=True, help='use shift for models')
parser.add_argument('--shift_div', default=8, type=int, help='number of div for shift (default: 8)')
parser.add_argument('--shift_place', default='blockres', type=str, help='place for shift (default: stageres)')

parser.add_argument('--temporal_pool', default=False, action="store_true", help='add temporal pooling')
parser.add_argument('--non_local', default=False, action="store_true", help='add non local block')

parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')

