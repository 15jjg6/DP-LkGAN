import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import os
import PIL
import time
import math
import sys

from tensorflow.keras import layers
from IPython import display

import tensorflow as tf
import pickle


class DP_LkGAN: 
    def __init__(self,
                    buffer_size,
                    batch_size,
                    max_fid_test_size,
                    desired_digit,
                    alpha,
                    beta,
                    gamma,
                    k,
                    c_val,
                    sigma,
                    epochs=100,
                    noise_dim=100,
                    num_examples_to_generate=128):

        # we should probably figure out what style we want to use... all caps, no caps...
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.max_fid_test_size = max_fid_test_size
        self.desired_digit = desired_digit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.c_val = c_val
        self.sigma = sigma
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate
        
        # generate seed value
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

        # Record and save FID scores
        self.fid_df = []


    # CURRENTLY NOT USING THIS
    def calculate_epsilon(self, c_val, sigma, epochs):
        L = self.BATCH_SIZE
        N = len(train_images)
        q = L / N
        delta = 0.0001 #10^5
        T = len(train_dataset) #number of steps

        return (q*math.sqrt(T*math.log(1/delta,10)))/self.sigma

    # CURRENTLY NOT USING THIS
    def calculate_clipping(self, sigma):
        # dan and Sean
        pass


    def calculate_fid(self):
        fake_images = self.generator(tf.random.normal([self.max_fid_test_size, self.noise_dim]))
        fake_images = fake_images.numpy()
        fake_images = fake_images.reshape(fake_images.shape[0], 28*28).astype('float32')
        fake_images = (fake_images * 127.5 + 127.5) / 255.0
        
        fake_mu = fake_images.mean(axis=0)
        fake_sigma = np.cov(np.transpose(fake_images))
        
        covSqrt = sp.linalg.sqrtm(np.matmul(fake_sigma, self.real_sigma))
        
        if np.iscomplexobj(covSqrt):
            covSqrt = covSqrt.real
        
        # might have to replace the norm below with just the sum of difference squared.
        fidScore = np.linalg.norm(self.real_mu - fake_mu) + np.trace(self.real_sigma + fake_sigma - 2 * covSqrt)
        self.fid_df.append(fidScore)
        return fidScore


    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)
        
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, 7, 7, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 14, 14, 64)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 28, 28, 1)

        return model


    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model


    def dis_loss_wrapper(self, beta, alpha):
        def dis_loss_def(real_output, fake_output):
                a = tf.math.reduce_mean(tf.math.pow(real_output - beta, 2.0 * tf.ones_like(real_output)))
                b = tf.math.reduce_mean(tf.math.pow(fake_output - alpha, 2.0 * tf.ones_like(fake_output)))
                return 1/2.0 * (a + b)

        return dis_loss_def 


    def gen_loss_wrapper(self, gamma, k):
        def gen_loss_def(fake_output):
            return tf.math.reduce_mean(tf.math.pow(tf.math.abs(fake_output - gamma),
                                                    k * tf.ones_like(fake_output)))
        return gen_loss_def


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.gen_loss_wrapper(self.gamma, self.k)(fake_output) 
            disc_loss = self.dis_loss_wrapper(self.beta, self.alpha)(real_output, fake_output) 

        # Compute the gradients for both loss functions
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables) # TODO

        # This seems pretty messy... should try to fix this / do it more efficiently
        for i, disc_layer_grads in enumerate(gradients_of_discriminator):
            ones_tensor   = tf.ones_like(disc_layer_grads)
            noise_tensor  = tf.random.normal(disc_layer_grads.shape, mean=0.0, stddev=self.sigma)
            C_tensor      = tf.ones_like(disc_layer_grads)*self.c_val

            disc_layer_grads = tf.math.truediv(disc_layer_grads, tf.math.maximum(ones_tensor, tf.math.truediv(tf.norm(disc_layer_grads,ord=2), C_tensor)))
            disc_layer_grads = tf.math.add(disc_layer_grads, noise_tensor)

            gradients_of_discriminator[i] = disc_layer_grads

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def train(self):

        print("\n__STARTING TRAINING__")

        # make sure this fid exit function works
        recent_lowest = 0
        fid_min = sys.maxsize
        
        start_time = time.time()
        for epoch in range(self.EPOCHS):
            start = time.time()

            for image_batch in self.train_dataset:
                self.train_step(image_batch)
        
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            display.clear_output(wait=True)
            print (f'Epoch {epoch+1} Completed...\n' +
                f'Total Runtime is {round(time.time()-start_time,2)}\n' +
                f'The FID score is: {self.calculate_fid()}\n')

            # make sure this fid exit function works
            # Training exit condition
            if self.fid_df[-1] < fid_min:
                fid_min = self.fid_df[-1]
                recent_lowest = 0
            else:
                recent_lowest += 1
            
            if recent_lowest >= 5:
                break
        
        predictions = self.generator(self.seed, training=False)
        output_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
        pickle.dump(predictions, open( f"gan_outputs/{output_string}​​​​​.p", "wb" ))

        print (f'__TRAINING COMPLETE__\nSummary of data:\nSigma: {self.sigma}\n' + 
                f'C:     {self.c_val}\nNumber of Epochs: {self.EPOCHS}\nAverage Epoch ' + 
                f'Runtime is {round((time.time()-start_time)/self.EPOCHS,2)} sec\n' + 
                f'Total Runtime is {round(time.time()-start_time,2)} sec\n' +
                f'The Final FID score is: {self.calculate_fid()}\n________________\n')
        
        # save FID scores as csv
        pd.DataFrame({'FID Scores':self.fid_df}).to_csv(f'fid_outputs/{output_string}.csv')


    def fid_setup(self,train_images):
        # Setup the FID calculations
        fid_train_images = train_images[:self.max_fid_test_size]
        fid_train_images = fid_train_images.reshape(fid_train_images.shape[0], 28 * 28).astype('float64')
        fid_train_images = fid_train_images / 255.0
        self.real_mu = fid_train_images.mean(axis=0)

        fid_train_images = np.transpose(fid_train_images)
        self.real_sigma = np.cov(fid_train_images)


    def train_gan(self,train_images):

        self.fid_setup(train_images)
        
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = (train_images - 127.5) / 127.5
        self.train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

        # check to make sure you understand this 
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                                discriminator_optimizer=self.discriminator_optimizer,
                                                generator=self.generator,
                                                discriminator=self.discriminator)

        self.train()


    def plot_fid(self):
        output_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
        final_fid_df = pd.read_csv(f'fid_outputs/{output_string}.csv')
        plt.figure()
        plt.plot(final_fid_df['FID Scores'])
        plt.xlabel("Epochs")
        plt.ylabel("FID Score")
        plt.show()