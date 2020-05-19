# midv-500-models
The repository contains a model for binary semantic segmentation of the documents.

## Dataset
Model is trained on [MIDV-500: A Dataset for Identity Documents Analysis and Recognition on Mobile Devices in Video Stream](https://arxiv.org/abs/1807.05786).

## Preparation

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
python midv-500-models/preprocess_data.py -i <input_folder> \
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
