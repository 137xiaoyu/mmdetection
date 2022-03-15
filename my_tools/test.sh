mkdir -p results/

CUDA_VISIBLE_DEVICES=0 python tools/test.py work_dirs/swin_b_same_norm/swin_b.py work_dirs/swin_b_same_norm/epoch_7.pth --out results/out_swin_b_same_norm_7.pkl
