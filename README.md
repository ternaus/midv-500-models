# midv-500-models
The repository contains a model for binary semantic segmentation of the documents.

![](https://habrastorage.org/webt/gy/-t/xn/gy-txnzezlnurcwwlv7q5vs77x4.jpeg)

* **Left**: input.
* **Center**: prediction.
* **Right**: overlay of the image and predicted mask.


## Installation

`pip install -U midv500models`

### Example inference

Jupyter notebook with an example: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lNv88MJOKgc-50XeYcHlJODpvT2JF9ru?usp=sharing)

## Dataset
Model is trained on [MIDV-500: A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream](https://arxiv.org/abs/1807.05786).

### Preparation

Download the dataset from the ftp server with
```bash
wget -r ftp://smartengines.com/midv-500/
```

Unpack the dataset
```bash
cd smartengines.com/midv-500/dataset/
unzip \*.zip
```

The resulting folder structure will be

```bash
smartengines.com
    midv-500
        dataset
            01_alb_id
                ground_truth
                    CA
                        CA01_01.tif
                    ...
                images
                    CA
                        CA01_01.json
                    ...
                ...
            ...
        ...
    ...
```

To preprocess the data use the script
```python
python midv500models/preprocess_data.py -i <input_folder> \
                                          -o <output_folder>
```

where `input_folder` corresponds to the file with the unpacked dataset and output folder will look as:

```bash
images
    CA01_01.jpg
    ...
masks
    CA01_01.png
```

target binary masks will have values \[0, 255\], where 0 is background and 255 is the document.

## Training

```bash
python midv500models/train.py -c midv500models/configs/2020-05-19.yaml \
                              -i <path to train>
```

## Inference

```bash
python midv500models/inference.py -c midv500models/configs/2020-05-19.yaml \
                                  -i <path to images> \
                                  -o <path to save preidctions>
                                  -w <path to weights>
```

## Weights
Unet with Resnet34 backbone: [Config](midv500models/configs/2020-05-19.yaml) [Weights](Unet_Resnet34.pth)
