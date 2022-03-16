mkdir -p results/

CUDA_VISIBLE_DEVICES=1 python tools/test.py work_dirs/swin_b_repeat_no_norm/swin_b_repeat_no_norm.py work_dirs/swin_b_repeat_no_norm/epoch_7.pth --out results/out_swin_b_repeat_no_norm_7.pkl
