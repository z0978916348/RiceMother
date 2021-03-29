#!/bin/bash
counter=1
while [ $counter -gt 0 ]; do
    # counter=`expr $counter + 1`
    python train_net.py --num-gpus 1 --config-file ../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x_c.yaml
    # echo $counter

    if [ "$?" -eq "0" ]; then
        echo "Success"
        break
    else
        echo "Failure"
    fi
done