CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/swin/swin_b_ms_1120_1280.py --cfg-options model.pretrained=swin_base_patch4_window7_224_22k.pth --work-dir work_dirs/swin_b_ms_1120_1280_pretrained_win7/

CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/swin/swin_b_repeat_autoaug_v2_ms_480_720.py --resume cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_repeat_autoaug_v2_ms_480_720/
