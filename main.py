import DP_LkGAN as dpgan
import tensorflow as tf


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5


# desired_digit = 1
# desired_train_images = train_images[train_labels == desired_digit]
# BUFFER_SIZE = len(desired_train_images)  # BUFFER_SIZE = 60000
# BATCH_SIZE = 256

# # gan_instance = dpgan.DP_LkGAN(desired_train_images, train_labels, batch_size=256, buffer_size=BUFFER_SIZE)
# gan_instance = dpgan.DP_LkGAN()
# gan_instance.train_gan(train_images=desired_train_images,
#                             buffer_size=BUFFER_SIZE,
#                             batch_size=BATCH_SIZE,
#                             desired_digit=desired_digit,
#                             alpha=0.5,
#                             beta=0.5,
#                             gamma=1,
#                             k=2,
#                             c_val=1,
#                             sigma=0.1,
#                             epochs=2,
#                             noise_dim=100,
#                             num_examples_to_generate=128)





alpha, beta, gamma, k = 0.5, 0.5, 1, 1.6
epochs, noise_dim, num_examples_to_generate, BATCH_SIZE = 1, 100, 128, 256

desired_digit_values = [0,1]
c_values = [10]
sigma_values = [0.01,0.05,0.1]

for desired_digit in desired_digit_values:
    for c_val in c_values:
        for sigma in sigma_values:
            desired_train_images = train_images[train_labels == desired_digit]
            BUFFER_SIZE = len(desired_train_images)

            gan_instance = dpgan.DP_LkGAN()
            gan_instance.train_gan(train_images=desired_train_images,
                                        buffer_size=BUFFER_SIZE,
                                        batch_size=BATCH_SIZE,
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
            
            # Compute FID scores...



desired_pairs = [[1,7],[0,8],[3,8],[4,9],[5,6]]

for pair in desired_pairs:
    # Call compare_gan_outputs/experiment
        # read pair[0] gan outputs
        # read pair[1] outputs
        # train classifier
        # compare scores ...