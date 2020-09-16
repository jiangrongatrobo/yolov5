###########################################
# best: epoch 152, mAP@50: 0.732, experiment: ./runs/exp5, apply Hardswish!!
nohup python -u train.py \
--weights yolov5s.pt \
--cfg ./models/yolov5s.yaml \
--data ./data/baiguang.yaml \
--device 0 > logs/yolov5s-training.log 2>&1 &

###########################################

# teacher model ??
# 
nohup python -u train.py \
--weights yolov5m.pt \
--workers 8 \
--cfg ./models/yolov5m.yaml \
--data ./data/baiguang.yaml \
--device 0,1 > logs/yolov5m-training.log 2>&1 &

###########################################

nohup python -u train.py \
--weights '' \
--cfg ./models/yolov5s-roborock.yaml \
--data ./data/coco.yaml \
--epochs 40 \
--device 0 > logs/yolov5s-roborock-pretraining.log 2>&1 &

nohup python -u train.py \
--weights yolov5s.pt \
--cfg ./models/yolov5s-roborock.yaml \
--data ./data/baiguang.yaml \
--device 0 > logs/yolov5s-roborock-training.log 2>&1 &

###############################################

nohup python -u train.py \
--weights yolov5m.pt \
--cfg ./models/yolov5m-roborock.yaml \
--data ./data/baiguang.yaml \
--device 0 > logs/yolov5m-roborock-training.log 2>&1 &

###############################################
python test.py \
--weights ./runs/exp5/weights/best.pt \
--data ./data/baiguang.yaml \
--device 1



nohup python -u train.py \
--weights yolov5m.pt \
--cfg ./models/yolov5m-roborock.yaml \
--data ./data/baiguang.yaml \
--device 1 > yolov5m-roborock-training.log 2>&1 &

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

