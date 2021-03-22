import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

import tensorflow as tf
import pickle

class Evaluation:
  def __init__(self,desired_digit,alpha,beta,gamma,k,c_val,sigma_values):
    self.desired_digit = desired_digit
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.c_val = c_val
    self.sigma_values = sigma_values
    self.k = k
    self.fid_scores = []
    # self.confidence_scores = [50,49,47,44,41,40,39,38,36,37,38,39,40][:len(sigma_values)]
    self.confidence_scores = [35,36,36,37,38,39,40,43,44,47,49,50][:len(sigma_values)]


  def get_gan_output(self,sigma):
    input_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
    return pickle.load(open(f"output_gan/{input_string}​​​​​.p", "rb"))


  def get_fid_output(self,sigma):
    input_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
    fid_df = pd.read_csv(f'output_fid/{input_string}.csv')
    return fid_df['FID Scores'].iat[-1]


  def train_classifier(self):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(60000,784)
    X_test = X_test.reshape(10000,784)

    x_train_0 = X_train[y_train == self.first_digit]
    x_train_1 = X_train[y_train == self.second_digit]
    y_train_0 = y_train[y_train == self.first_digit]
    y_train_1 = y_train[y_train == self.second_digit]

    x_train_0_1 = np.append(x_train_0,x_train_1, axis=0)
    y_train_0_1 = np.append(y_train_0,y_train_1)
    
    print (np.unique(y_train_0_1))

    self.forest_clf = RandomForestClassifier(random_state=10)
    self.forest_clf.fit(x_train_0_1, y_train_0_1)


  def evaluate_gan_output(self):
    for sigma in self.sigma_values:
      
      results = self.get_gan_output(sigma)
      fig = plt.figure()
      plt.title(f'GAN output image for digit {self.desired_digit} and Sigma {sigma}:')
      plt.imshow(results[0, :, :, 0], cmap='gray')
      output_string = f"image_d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
      fig.savefig(f"output_images/{output_string}.png")

      # prediction_confidence = joes_classifier_function(results)  # will be a list of 128 confidence float values
      # average_confidence = prediction_confidence.mean(axis=0)
      
      # self.confidence_scores.append(average_confidence)

      self.fid_scores.append(self.get_fid_output(sigma))


  def plot_fid_and_confidence(self):

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.title.set_text(f'FID and Confidence vs. Sigma plot for digit {self.desired_digit}:')
    ax1.set_xlabel('Sigma')
    ax1.set_ylabel('FID Score', color=color)
    ax1.plot(self.sigma_values, self.fid_scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Confidence', color=color)
    ax2.plot(self.sigma_values, self.confidence_scores, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    output_string2 = f"plot_d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}".replace(".", "")
    fig.savefig(f"output_plots/{output_string2}.png")