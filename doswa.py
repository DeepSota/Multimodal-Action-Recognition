#代码块
import torch

output_path = '/data2/wsp/multi-modal-tsm/checkpoint/TSM_mmvpr_RTD_resnet101_shift8_blockres_avg_segment8_e30_best/'
#output_path = './output/convnextv2_large-384/'

def do_swa():
    skip = ['num_batches_tracked']#['relative_position_index', 'num_batches_tracked']
    #    checkpoint = ['./swa/pvt_a3_lr4_train_self_crop_1024_512_256x256-epoch=00-val_miou_epoch=0.690-0.8112.pth','./swa/pvt_a5_1024_1024_256x256_self-epoch=11-val_miou_epoch=0.696-0.8124.pth','./swa/pvt_ssr_1024_512_128x128_self-epoch=16-val_miou_epoch=0.726-0.8056.pth']
    checkpoint = [
    f'{output_path}ckpt_27.pth.tar',
    f'{output_path}ckpt_best.pth.tar',
    #f'{output_path}10.pth',
    #f'{output_path}14.pth',
    ]

    K = len(checkpoint)
    swa = None
    for k in range(K):
        state_dict = torch.load(checkpoint[k], map_location=lambda storage, loc: storage)['state_dict']#['state_dict'] ['net']
        if swa is None:
            swa = state_dict
        else:
            for k, v in state_dict.items():
                # print(k)
                if any(s in k for s in skip): continue
                swa[k] += v

    for k, v in swa.items():
        if any(s in k for s in skip): continue
        print(k,'k')
        swa[k] /= K
    pth = {}
    pth['state_dict'] = swa
    #不添加版本，可能会出错
    pth['pytorch-lightning_version']='1.6.5'
    for key in state_dict.keys():
        print(key)
    print('----------------')
    for key in pth['state_dict'].keys():
        print(key)
    torch.save(pth, f'{output_path}ckpt_swa.pth.tar')
    return swa

do_swa()
