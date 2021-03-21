import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

import tensorflow as tf
import pickle

class Evaluation:
  def __init__(self,pair,alpha,beta,gamma,k,c_val,sigma):
    self.first_digit = pair[0]
    self.second_digit = pair[1]
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.c_val = c_val
    self.sigma = sigma
    self.k = k


  def get_gan_output(self,desired_digit):
    output_string = f"d{desired_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
    return pickle.load(open(f"gan_outputs/{output_string}​​​​​.p", "rb"))


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

    results_for_0 = self.get_gan_output(self.second_digit)
    plt.imshow(results_for_0[0, :, :, 0], cmap='gray')
    results_for_1 = self.get_gan_output(self.first_digit)
    plt.imshow(results_for_1[0, :, :, 0], cmap='gray')

    results_for_0 = results_for_0.numpy().reshape(128, 784).astype('float64')
    results_for_1 = results_for_1.numpy().reshape(128, 784).astype('float64')

    results = np.append(results_for_0, results_for_1, axis=0)
 
    prediction = self.forest_clf.predict_proba(results)

    avg_prediction_1 = prediction[:128].mean(axis=0)
    avg_prediction_0 = prediction[128:].mean(axis=0)

    # ground_truth = [1] * 128
    # ground_truth.extend([1] * 128)

    for i in prediction: print(i)

    # plt.plot(prediction)
    # plt.show()


  def plot_fid(self):
    output_string = f"d{self.first_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
    fid_df1 = pd.read_csv(f'fid_outputs/{output_string}.csv')
    plt.figure()
    output_string = f"d{self.second_digit}_a{self.alpha}_b{self.beta}_g{self.gamma}_k{self.k}_c{self.c_val}_s{self.sigma}".replace(".", "")
    fid_df2 = pd.read_csv(f'fid_outputs/{output_string}.csv')
    plt.figure()
    plt.plot(fid_df1['FID Scores'])
    plt.plot(fid_df2['FID Scores'])
    plt.xlabel("Epochs")
    plt.ylabel("FID Score")
    print(fid_df1)
    print(fid_df2)

    plt.show()