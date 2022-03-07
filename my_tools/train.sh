python tools/train.py configs/swin/swin_b_autoaug_v2_ms_768_1024.py --resume-from cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_autoaug_v2_ms_768_1024/

python tools/train.py configs/swin/swin_b_mstrain_480_800.py --resume-from cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_mstrain_480_800/

python tools/train.py configs/swin/swin_b_ms_1120_1280.py --resume-from cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_ms_1120_1280/

CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/swin/swin_b_ms_1120_1280.py --cfg-options model.pretrained=swin_base_patch4_window7_224_22k.pth --work-dir work_dirs/swin_b_ms_1120_1280_pretrained_win7/

CUDA_VISIBLE_DEVICES=2 python tools/train.py configs/swin/swin_b_ms_1120_1280.py --resume cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_ms_1120_1280_init_lr/
