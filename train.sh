CUDA_VISIBLE_DEVICES=3  python main.py  --arch resnet50 --num_segments 8  --gd 20 --lr 0.01 --lr_steps 10 20 --epochs 32 --batch-size 12 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=50  --shift True --shift_div=8 --shift_place=blockres --npb


# CUDA_VISIBLE_DEVICES=0  python main_mobliev2.py  --arch mobilenetv2 --num_segments 8  --gd 20 --lr 0.01 --lr_steps 10 20 --epochs 31 --batch-size 12 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=50  --shift True --shift_div=8 --shift_place=blockres --npb

