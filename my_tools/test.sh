mkdir -p results/

CUDA_VISIBLE_DEVICES=7 python tools/test.py work_dirs/swin_b_repeat_ms_720_960/swin_b_repeat_ms_720_960.py work_dirs/swin_b_repeat_ms_720_960/epoch_12.pth --out results/swin_b_repeat_ms_720_960.pkl
