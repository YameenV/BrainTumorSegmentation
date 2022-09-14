import nibabel as nib
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


flairFilename = tf.data.Dataset.list_files("../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii", shuffle=False, seed=42)
t2Filename = tf.data.Dataset.list_files("../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii", shuffle=False, seed=42)
t1ceFilename = tf.data.Dataset.list_files("../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii", shuffle=False, seed=42)
maskFilename = tf.data.Dataset.list_files("../input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii", shuffle=False, seed=42)

def dataLoader():

    for flair, t2, t1ce, mask in tf.data.Dataset.zip((flairFilename, t2Filename, t1ceFilename, maskFilename)).as_numpy_iterator():
        tempFlair = nib.load(flair.decode("utf-8")).get_fdata()
        tempFlairScal = scaler.fit_transform(tempFlair.reshape(-1, 1)).reshape(tempFlair.shape)

        tempT2 = nib.load(t2.decode("utf-8")).get_fdata()
        tempT2Scal = scaler.fit_transform(tempT2.reshape(-1, 1)).reshape(tempFlair.shape)

        tempT1ce = nib.load(t1ce.decode("utf-8")).get_fdata()
        tempT1ceScal = scaler.fit_transform(tempT1ce.reshape(-1, 1)).reshape(tempFlair.shape)

        tempMask = nib.load(mask.decode("utf-8")).get_fdata()
        tempMask=tempMask.astype(np.uint8)
        tempMask[tempMask==4] = 3

        _, count = np.unique(tempMask, return_counts=True)

        if (1 - (count[0] / count.sum())) > 0.01:
            tempTraget = np.stack([tempFlairScal, tempT2Scal, tempT1ceScal], axis=3)
            tempTragetCrop = tempTraget[56:184, 56:184, 13:141]
            tempMaskCrop = tempMask[56:184, 56:184, 13:141]
            
            yCategorical = to_categorical(tempMaskCrop,num_classes = 4)


            yield (tempTragetCrop, yCategorical)



trainDataset = tf.data.Dataset.from_generator(
    dataLoader,
    output_types = (tf.float32, tf.float32),
    output_shapes = ((128, 128, 128, 3),(128, 128, 128, 4))
)


