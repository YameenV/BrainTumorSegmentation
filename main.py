from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf
from unet3d import unet3d
from metrics import iou, diceCoef, diceLoss
from datapipline import  trainDataset

inputs = (128,128, 128, 3)
model = unet3d(inputs)
metrics = [diceCoef, iou, Recall(), Precision()]

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss = diceLoss,
    metrics = metrics
)

model.fit(
    trainDataset.batch(1),
    epochs=10,
    max_queue_size=1,
)

model.save("./model.h5")
