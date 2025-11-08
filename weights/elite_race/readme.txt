使用Ultimate8进行图像处理

使用Ultimate_Ready进行训练


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    242/300      21.6G     0.1061    0.09759     0.8029          4        640: 100%|██████████| 401/401 [02:13<00:00,  3
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:13<
                   all      10799      12152      0.969      0.968       0.98      0.978
EarlyStopping: Training stopped early as no improvement observed in last 100 epochs. Best results observed at epoch 142, best model saved as best.pt.
To update EarlyStopping(patience=100) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

242 epochs completed in 9.994 hours.
Optimizer stripped from runs/detect/escherichia_train2/weights/last.pt, 5.5MB
Optimizer stripped from runs/detect/escherichia_train2/weights/best.pt, 5.5MB

Validating runs/detect/escherichia_train2/weights/best.pt...
Ultralytics 8.3.170 🚀 Python-3.8.20 torch-2.1.0+cu121 CUDA:0 (NVIDIA GeForce RTX 4090 D, 24210MiB)
YOLOv12n summary (fused): 159 layers, 2,557,118 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 50/50 [00:26<
                   all      10799      12152       0.98      0.965      0.983      0.981
                     H       5616       5942      0.978      0.966      0.983      0.981
                target       5847       6210      0.983      0.965      0.983      0.981
Speed: 0.1ms preprocess, 0.4ms inference, 0.0ms loss, 1.1ms postprocess per image
Results saved to runs/detect/escherichia_train2