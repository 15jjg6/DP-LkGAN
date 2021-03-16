import DP_LkGAN as dpgan
import Evaluation as gan_eval
import tensorflow as tf


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

alpha, beta, gamma, k = 0.5, 0.5, 1, 1.6
epochs, noise_dim, num_examples_to_generate, batch_size, max_fid_test_size = 10, 100, 128, 256, 10000

desired_digit_values = [1,7]
c_values = [10]
sigma_values = [0.05]

# TODO: Try running a GridSearch here...
for desired_digit in desired_digit_values:
    for c_val in c_values:
        for sigma in sigma_values:
            desired_train_images = train_images[train_labels == desired_digit]
            buffer_size = len(desired_train_images)

            gan_instance = dpgan.DP_LkGAN(buffer_size=buffer_size,
                                            batch_size=batch_size,
                                            max_fid_test_size=max_fid_test_size,
                                            desired_digit=desired_digit,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            k=k,
                                            c_val=c_val,
                                            sigma=sigma,
                                            epochs=epochs,
                                            noise_dim=noise_dim,
                                            num_examples_to_generate=num_examples_to_generate)

            gan_instance.train_gan(train_images=desired_train_images)
            gan_instance.plot_fid()


desired_pairs = [[1,7]]

for pair in desired_pairs:
    for c_val in c_values:
        for sigma in sigma_values:
            evaluator = gan_eval.Evaluation(pair=pair,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            k=k,
                                            c_val=c_val,
                                            sigma=sigma)
            evaluator.train_classifier()
            evaluator.evaluate_gan_output()
            evaluator.plot_fid()

            # TODO: plot/print the prediction confidence values
            # TODO: Plot ROC_AUC curve...
            # TODO: plot FID (end of training) vs confidence over multiple sigma values.
            # TODO: Figure out how to have multiple plots plotted separately (currently they all overlap)