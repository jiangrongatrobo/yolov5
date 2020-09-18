###########################################
# best: epoch 152, mAP@50: 0.732, experiment: ./runs/exp5, apply Hardswish!!
nohup python -u train.py \
--weights weights/yolov5s.pt \
--cfg ./models/yolov5s.yaml \
--data ./data/baiguang.yaml \
--device 0 > logs/yolov5s-training.log 2>&1 &

###########################################

# mAP@50: 0.772
nohup python -u train.py \
--weights weights/yolov5m.pt \
--workers 8 \
--cfg ./models/yolov5m.yaml \
--data ./data/baiguang.yaml \
--device 0,1 > logs/yolov5m-training.log 2>&1 &

python test.py \
--weights /workspace/yolov5-v3/yolov5/runs/exp122/weights/best.pt \
--data ./data/baiguang.yaml \
--device 1 \
--conf-thres 0.2

# python -u train.py \
# --weights /workspace/yolov5-v3/yolov5/runs/exp122/weights/best.pt \
# --workers 8 \
# --cfg ./models/yolov5m.yaml \
# --data ./data/baiguang.yaml \
# --batch-size 1 \
# --device 0,1

###########################################

nohup python -u train.py \
--weights '' \
--cfg ./models/yolov5s-roborock.yaml \
--data ./data/coco.yaml \
--epochs 200 \
--workers 16 \
--batch-size 32 \
--device 0,1 > logs/yolov5s-roborock-pretraining.log 2>&1 &
# trained 40 epoch

nohup python -u train.py \
--weights './runs/exp194/weights/best.pt' \
--cfg ./models/yolov5s-roborock.yaml \
--data ./data/baiguang.yaml \
--workers 16 \
--batch-size 32 \
--device 0,1 > logs/yolov5s-roborock-training.log 2>&1 &

###############################################
python train.py \
--cfg ./models/yolov5-roborock-nas.yaml \
--data ./data/coco.yaml \
--device 0,1 \
--workers 16 \
--epochs 200 \
--batch-size 16 \
--nas \
--nas-stage 0

python train.py \
--cfg ./models/yolov5-roborock-nas.yaml \
--data ./data/baiguang.yaml \
--device 0,1 \
--workers 16 \
--epochs 200 \
--batch-size 16 \
--nas \
--nas-stage 0

python train.py \
--cfg ./models/yolov5-roborock-nas.yaml \
--data ./data/baiguang.yaml \
--weights ./runs/exp241/weights/last.pt \
--device 0,1 \
--workers 16 \
--epochs 200 \
--batch-size 4 \
--nas \
--nas-stage 1

###############################################





nohup python -u train.py \
--weights weights/yolov5m.pt \
--cfg ./models/yolov5m-roborock.yaml \
--data ./data/baiguang.yaml \
--device 1 > yolov5m-roborock-training.log 2>&1 &

