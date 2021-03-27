import DP_LkGAN as dpgan
import Evaluation as gan_eval
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


# Sean: 3,8
# Dan: 2,9
# Tania: 1,7
# Joe: 4,5
# Alex: 0,6

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

alpha  = 0.5
beta   = 0.5
gamma  = 1
c_val  = 10

epochs                   = 150
noise_dim                = 100
num_examples_to_generate = 128
batch_size               = 256
max_fid_test_size        = 10000

k_values             = [1, 1.5, 2.0, 2,5, 3.0]
desired_digit_values = [0, 6]
sigma_values         = [0, 0.005, 0.01361, 0.02819, 0.03431, 0.04110, 0.05]


for k in k_values:
    for desired_digit in desired_digit_values:
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
            gan_instance.save_plot_fid()


for k in k_values:
    for desired_digit in desired_digit_values:
        evaluator = gan_eval.Evaluation(desired_digit=desired_digit,
                                        alpha=alpha,
                                        beta=beta,
                                        gamma=gamma,
                                        k=k,
                                        c_val=c_val,
                                        sigma_values=sigma_values)
        evaluator.evaluate_gan_output()
        evaluator.plot_fid_and_confidence()


fid_all_digits = pd.DataFrame()
confidence_all_digits = pd.DataFrame()
digits_colors = ['tab:blue','tab:red','tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

for k in k_values:

    fig, ax1 = plt.subplots()
    ax1.title.set_text(f'FID and Confidence vs. Sigma plot for all digits (k={k}):')
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('FID Score')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Confidence')

    for desired_digit in desired_digit_values:

        input_string = f"table_d{desired_digit}_a{alpha}_b{beta}_g{gamma}_k{k}_c{c_val}".replace(".", "")
        fid_and_confidence_df = pd.read_csv(f'output_fid_conf_tables/{input_string}.csv')
        print(fid_and_confidence_df)
        fid_all_digits[desired_digit] = fid_and_confidence_df['FID Scores']
        confidence_all_digits[desired_digit] = fid_and_confidence_df['Confidence']

        color = digits_colors[desired_digit]
        ax1.plot(sigma_values, fid_all_digits[desired_digit], color=color)
        ax1.tick_params(axis='y')

        ax2.plot(sigma_values, confidence_all_digits[desired_digit], color=color,linestyle='dashed')
        ax2.tick_params(axis='y')

    output_string = f"a{alpha}_b{beta}_g{gamma}_k{k}_c{c_val}".replace(".", "")

    fig.savefig(f"output_merged/plot_separated_{output_string}.png")

    average_fid = fid_all_digits.mean(axis=1)
    average_conf = confidence_all_digits.mean(axis=1)

    fig2, ax21 = plt.subplots()
    color = 'tab:red'
    ax21.title.set_text(f'Average FID and Confidence vs. Sigma plot (k={k}):')
    ax21.set_xlabel('Sigma')
    ax21.set_ylabel('FID Score', color=color)
    ax21.plot(sigma_values, average_fid, color=color)
    ax21.tick_params(axis='y', labelcolor=color)
    
    ax22 = ax21.twinx()

    color = 'tab:blue'
    ax22.set_ylabel('Confidence', color=color)
    ax22.plot(sigma_values, average_conf, color=color)
    ax22.tick_params(axis='y', labelcolor=color)

    fig2.savefig(f"output_merged/plot_averaged_{output_string}.png")
    pd.DataFrame({'Sigma':sigma_values, 'FID Scores':average_fid, 'Confidence':average_conf}).to_csv(f'output_merged/table_average_{output_string}.csv')
    plt.close('all')

# cd documents/github/dp-lkgan
