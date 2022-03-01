python -m debugpy --listen localhost:5678 --wait-for-client my_tools/make_submission.py D:/137/dataset/competitions/shipdet_dcic/xmy_data/test.txt my_tools/results/out.pkl my_tools/results/submission.csv --score_thrs 0.85 0.8

python my_tools/make_submission.py D:/137/dataset/competitions/shipdet_dcic/xmy_data/test.txt my_tools/results/out_21.pkl my_tools/results/submission_21.csv --score_thrs 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3

python my_tools/make_submission.py D:/137/dataset/competitions/shipdet_dcic/xmy_data/test.txt my_tools/results/out_36.pkl my_tools/results/submission_36.csv --score_thrs 0.85 0.8 0.75 0.7 0.65 0.6 0.55 0.5 0.45 0.4 0.35 0.3
