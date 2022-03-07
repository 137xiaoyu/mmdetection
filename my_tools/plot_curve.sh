python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b/20220224_194225.log.json --keys loss --legend loss

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b/20220224_194225.log.json --keys mAP --legend mAP

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_autoaug_v2_ms_768_1024/20220225_175456.log.json --keys loss --legend loss

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_autoaug_v2_ms_768_1024/20220225_175456.log.json --keys mAP --legend mAP

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_mstrain_480_800/20220301_193246.log.json --keys loss --legend loss

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_mstrain_480_800/20220301_193246.log.json --keys mAP --legend mAP

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_ms_1120_1280/20220304_145805.log.json --keys loss --legend loss

python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/swin_b_ms_1120_1280/20220304_145805.log.json --keys mAP --legend mAP
