# YoloV3
A minimal PyTorch implementation of YOLOv3, with support for inference. 
This is a wrapper of YOLOV3-pytorch implementation [here](https://github.com/eriklindernoren/PyTorch-YOLOv3) 
as a standalone python package.

## Install
    
    $ pip install git+https://github.com/mkocabas/yolov3-pytorch.git
    
## Performance

| Model                   | mAP (min. 50 IoU) |
| ----------------------- |:-----------------:|
| YOLOv3 608 (paper)      | 57.9              |
| YOLOv3 608 (this impl.) | 57.3              |
| YOLOv3 416 (paper)      | 55.3              |
| YOLOv3 416 (this impl.) | 55.5              |

## Inference
Uses pretrained weights to make predictions on images.

    $ python examples/detect.py --image_folder data/samples/

Below table displays the inference times when using as inputs images scaled to 256x256.

| Backbone                | GPU      | FPS      |
| ----------------------- |:--------:|:--------:|
| ResNet-101              | Titan X  | 53       |
| ResNet-152              | Titan X  | 37       |
| Darknet-53 (paper)      | Titan X  | 76       |
| Darknet-53 (this impl.) | 1080ti   | 74       |

## Sample usage as a package

```python
from yolov3.yolo import YOLOv3
from torchvision.transforms import Compose, ToTensor, Normalize

detector = YOLOv3(device='cuda', img_size=608)
preprocess_img = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
])

img = preprocess_img('data/sample.png').to('cuda')

predictions = detector(img)

```

## Credit

- YOLOv3: An Incremental Improvement

_Joseph Redmon, Ali Farhadi_ <br>

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

- YOLOv3 PyTorch implementation: [link](https://github.com/eriklindernoren/PyTorch-YOLOv3)

