mkdir -p my_tools/results/

python tools/test.py work_dirs/swin_b_autoaug_v2_ms_768_1024/swin_b_autoaug_v2_ms_768_1024.py work_dirs/swin_b_autoaug_v2_ms_768_1024/epoch_21.pth --out my_tools/results/out_swin_b_autoaug_v2_ms_768_1024_21.pkl

python tools/test.py work_dirs/swin_b/swin_b.py work_dirs/swin_b/epoch_7.pth --out my_tools/results/out_swin_b_7.pkl

python tools/test.py work_dirs/swin_b_ms_1120_1280/swin_b_ms_1120_1280.py work_dirs/swin_b_ms_1120_1280/epoch_12.pth --out my_tools/results/out_swin_b_ms_1120_1280.pkl

CUDA_VISIBLE_DEVICES=3 python tools/test.py work_dirs/swin_b_ms_1120_1280_init_lr/swin_b_ms_1120_1280.py work_dirs/swin_b_ms_1120_1280_init_lr/epoch_12.pth --out my_tools/results/out_swin_b_ms_1120_1280_init_lr_12.pkl
