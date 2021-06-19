# Waymo challenge 2021: motion prediction

[Motion Prediction](https://waymo.com/open/challenges/2021/motion-prediction/)
[CVPR2021 workshop](http://cvpr2021.wad.vision/)

## Dataset

Download
[datasets](https://console.cloud.google.com/storage/browser/waymo_open_dataset_motion_v_1_0_0)
`uncompressed/tf_example/{training,validation,testing}`

## Prerender

Change paths to input dataset and output folders

```bash
python prerender.py \
    --data /home/data/waymo/training \
    --out ./train
    
python prerender.py \
    --data /home/data/waymo/validation \
    --out ./dev \
    --n-shards 1
    
python prerender.py \
    --data /home/data/waymo/testing \
    --out ./test \
    --n-shards 1
```

## Training

```bash
MODEL_NAME=xception71
python train.py \
    --train-data ./train \
    --dev-data ./dev \
    --save ./${MODEL_NAME} \
    --model ${MODEL_NAME} \
    --img-res 224 \
    --in-channels 25 \
    --time-limit 80 \
    --n-traj 6 \
    --lr 0.001 \
    --batch-size 48 \
    --n-epochs 120
```

## Useful links

* [kaggle lyft 3rd place solution](https://gdude.de/blog/2021-02-05/Kaggle-Lyft-solution)
