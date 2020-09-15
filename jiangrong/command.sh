cd /workspace/yolov5-v3/yolov5

python train.py \
--weights yolov5s.pt \
--cfg ./models/yolov5s-roborock.yaml \
--data ./data/baiguang.yaml \
--device 0

nohup \
python train.py \
--cfg ./models/yolov5s-roborock-nas.yaml \
--data ./data/baiguang.yaml \
--device 0 \
--batch-size 8 \
--workers 8 \
--nas \
--nas-stage 1 \
&

# mAP@50 71.8
python test.py \
--weights ./runs/exp5/weights/last.pt \
--data ./data/baiguang.yaml \
--device 0
