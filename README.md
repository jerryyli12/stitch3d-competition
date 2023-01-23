# stitch3d-competition

This repo contains my submission for the Candelytics/Stitch3D ML competition.

## Methodology

I used the [P2P](https://arxiv.org/pdf/2208.02812.pdf) model, which is [open source](https://github.com/wangzy22/P2P) under the MIT License. The P2P model is, to my knowledge, the best performing model on the ScanObjectNN dataset with publicly available code.

I began with the P2P pretrained model, which is only trained and tested on the PB_T50_RS variant of the ScanObjectNN dataset. I trained an additional 30 epochs (atop the original 300) on all four perturbed variants of the ScanObjectNN training set. I did not use any additional datasets. For further technical details of the P2P model, please see the linked paper.

## Testing

```P2P/preds.csv``` contains my model's predictions on the test set. The first column contains the prediction, the second column contains the ground truth, and the third column contains the dataset from which the sample is from (1=PB_T25, 2=PB_T25_R, 3=PB_T50_R, 4=PB_T50_RS). 

My model achieves the following accuracies on the test set:
| PB_T25 | PB_T25_R | PB_T50_R | PB_T50_RS | Overall (competition evaluation metric) |
| ------ | -------- | -------- | --------- | ------ |
| 91.84519695922599 | 90.52886277220878 | 88.7080013855213 | 89.07009021512839 | **89.53072875207211** |

To run inference yourself, please first follow the [Installation Prerequisites](https://github.com/wangzy22/P2P#installation-prerequisites) section of the P2P documentation and ```pip install pandas```. Next, download the ScanObjectNN dataset as well as [my model weights](https://drive.google.com/file/d/1XMLYqNbrxKHlTOMvp2bhAO6NTyRsCpHA/view?usp=share_link). Structure the directory as follows:

```
P2P/
|-- data/
    |-- ScanObjectNN/
        |-- main_split/
            |-- training_objectdataset_augmented25_norot.h5
            |-- training_objectdataset_augmented25rot.h5
            |-- training_objectdataset_augmentedrot.h5
            |-- training_objectdataset_augmentedrot_scale75.h5
            |-- test_objectdataset_augmented25_norot.h5
            |-- test_objectdataset_augmented25rot.h5
            |-- test_objectdataset_augmentedrot.h5
            |-- test_objectdataset_augmentedrot_scale75.h5
|-- Exp/
    |-- ScanObjectNN/
        |-- 30ep/
            |-- model/
                |-- model_last.pth
```

Finally, run inference with the following command from the ```P2P``` directory. Note that, as configured, this requires a GPU with 16GB of memory. It takes ~1.5 hours on a V100 GPU. At the end, it will output ```preds.csv``` and print out the test set accuracy, which should match with the numbers provided above.
```
bash tool/test.sh 30ep config/ScanObjectNN/p2p_HorNet-L-22k-mlp.yaml ScanObjectNN
```
