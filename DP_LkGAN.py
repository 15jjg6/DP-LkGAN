import glob
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import PIL
import time

from tensorflow.keras import layers
from IPython import display

import tensorflow as tf
import pickle


class DP_LkGAN: 
    def __init__(self,
                    buffer_size,
                    batch_size,
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
        
        self.BUFFER_SIZE =buffer_size
        self.BATCH_SIZE = batch_size
        self.desired_digit = desired_digit
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.c_val = c_val
        self.sigma = sigma
        self.EPOCHS = epochs
        self.noise_dim = noise_dim
        self.num_examples_to_generate = num_examples_to_generate
        
        # generate seed value
        self.seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])

    # CURRENTLY NOT USING THIS
    def calculate_ep(self, C, sigma, batch_size, epochs):
        # dan and Sean 
        pass

    # CURRENTLY NOT USING THIS
    def calculate_clipping(self, sigma):
        # dan and Sean
        pass


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
        
        start_time = time.time()
        for epoch in range(self.EPOCHS):
            start = time.time()

            for image_batch in self.train_dataset:
                self.train_step(image_batch)
        
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.checkpoint_prefix)

            display.clear_output(wait=True)
            print (f'Epoch {epoch+1} Completed...' +
                f'\nTotal Runtime is {round(time.time()-start_time,2)}\n')
                # f'\nThe FID score is: {calculate_fid()}\n\n')

        # Generate after the final epoch
        # display.clear_output(wait=True)
        # generate_and_save_images(generator,
        #                          epochs,
        #                          seed)
        
        predictions = self.generator(self.seed, training=False)
        strings = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
        pickle.dump(predictions, open( f"gan_outputs/{strings}​​​​​.p", "wb" ))

        print (f'TRAINING COMPLETE\nSummary of data:\nSigma: {self.sigma}\n' + 
                f'C:     {self.c_val}\nNumber of Epochs: {self.EPOCHS}\nAverage Epoch ' + 
                f'Runtime is {round((time.time()-start_time)/self.EPOCHS,2)} sec\n' + 
                f'Total Runtime is {round(time.time()-start_time,2)} sec')
                # + f'\n\nThe Final FID score is: {calculate_fid()}')


    def train_gan(self,train_images):

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
        print("Starting Training")
        self.train()