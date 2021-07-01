from mmdet.apis import init_detector, inference_detector
import cv2

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'my_tools/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
result = inference_detector(model, 'demo/demo.jpg')

img_show = cv2.imread('demo/demo.jpg')

model.show_result(img_show,
                  result,
                  show=True,
                  out_file='work_dirs/demo.jpg',
                  score_thr=0.3)

print(1)
