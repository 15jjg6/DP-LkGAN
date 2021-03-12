import DP_LkGAN as gan
import tensorflow as tf


# loads MNIST training dataset, don't need the testing portion.
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5
train_images = train_images[train_labels == 1]


# BUFFER_SIZE = 60000
BUFFER_SIZE = len(train_images)
BATCH_SIZE = 256
print(len(train_images))


gan_exp = gan.DP_LkGAN(train_images, train_labels, BUFFER_SIZE=BUFFER_SIZE)
gan_exp.setup_experiment(0.5, 0.5, 1, 2, 1, 0.1)
