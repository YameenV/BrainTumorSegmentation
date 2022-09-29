# Brain Tumor Segmentation

## 3D Unet Architecture 
![alt text](./BrainTumor/unet3dArc.png)

## Dataset
### BraTS2020 Dataset
https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

## Results :- 
> Version 3 

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.94       | 0.89   | 0.98 | 0.98 |

![alt text](./BrainTumor/bratsV3_2.png)
![alt text](./BrainTumor/bratsV3_3.png)
![alt text](./BrainTumor/bratsV3_11.png)


> Version 2

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.88 | 0.79   | 0.97     | 0.97        |

![alt text](./BrainTumor/bratsV2_8.png)
![alt text](./BrainTumor/bratsV2_3.png)
![alt text](./BrainTumor/bratsV2_2.png)


> Version 1 

| DiceCoef   | IOU    | Recall   | Precision   |
|:----------:|:------:|:--------:|:-----------:|
| 0.80       | 0.68   | 0.78     | 0.82        |

![alt text](./BrainTumor/bratsV1_2.png)
![alt text](./BrainTumor/bratsV1_3.png)
![alt text](./BrainTumor/bratsV1_4.png)

## Lessons Learned
After analyzing various results from different version of the model, I should have used a Weighted Loss function as the dataset contain less number of samples from classes 2 and 3.  This lead to high IOU score but the model preform worst for predicting classes 2 and 3

## Reference

1. Ozg ̈un C ̧ i ̧cek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas
Brox, and Olaf Ronneberger (2016) 3D U-Net: Learning Dense Volumetric
Segmentation from Sparse Annotation Google DeepMind, London, UK, Computer Science Department, University of Freiburg, Germany

2. 3D-UNet Medical Image Segmentation for TensorFlow NVIDIA



