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
    self.confidence_scores = []


  def get_gan_output(self,sigma):
    input_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
    return pickle.load(open(f"output_gan/{input_string}​​​​​.p", "rb"))


  def get_fid_output(self,sigma):
    input_string = f"d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
    fid_df = pd.read_csv(f'output_bestfid/{input_string}.csv')
    return fid_df['FID Scores'].iat[-1]


  def get_predictions(self,results):
 
    with open("mnist_classifier_model.pkl", 'rb') as file:
      pickle_model = pickle.load(file)

    results = results.numpy().reshape(128, 784).astype('float64')
    
    predictions = pickle_model.predict_proba(results)
    predictions_for_val = [pred[self.desired_digit] for pred in predictions]
    confidence_in_digit = np.mean(predictions_for_val)

    return confidence_in_digit


  def evaluate_gan_output(self):
    for sigma in self.sigma_values:
      results = self.get_gan_output(sigma)
      fig = plt.figure()
      plt.title(f'GAN output image for digit {self.desired_digit} and Sigma {sigma}:')
      plt.imshow(results[0, :, :, 0], cmap='gray')
      output_string = f"image_d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{sigma}".replace(".", "")
      fig.savefig(f"output_images/{output_string}.png")

      average_confidence = self.get_predictions(results) 
      self.confidence_scores.append(average_confidence)

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
    fig.savefig(f"output_fid_conf_plots/{output_string2}.png")

    output_string3 = f"table_d{self.desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}".replace(".", "")
    pd.DataFrame({'Sigma':self.sigma_values, 'FID Scores':self.fid_scores, 'Confidence':self.confidence_scores}).to_csv(f'output_fid_conf_tables/{output_string3}.csv')