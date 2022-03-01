python tools/train.py configs/swin/swin_b_autoaug_v2_ms_768_1024.py --resume-from cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_autoaug_v2_ms_768_1024

python tools/train.py configs/swin/swin_b_mstrain_480_800.py --resume-from cascade_mask_rcnn_swin_base_patch4_window7.pth --work-dir work_dirs/swin_b_mstrain_480_800
