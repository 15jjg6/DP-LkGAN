# DP-LkGAN
## Producing Differentially Private Data using LkGAN

### Joe Grosso, Alex Le Blanc, Sean Kato, Tania Sidhom, Daniel Howells
#### 10/04/2021
Repository for Applied Mathematics and Engineering Final Year Thesis Project: Producing Differentially Private Data using LkGAN

## Abstract
>_Generative adversarial networks (GANs) have been a rising technique in addressing complex machine learning problems due to their strong theoretical foundation and ability to generate new data without changing the original statistical distribution. A downside of GANs is that they have the ability to unintentionally remember data points which can lead to an exposure of private or sensitive information. The  DP-LkGAN introduces an approach to address this issue. Its architecture is a modified GAN framework that combines the ideas of previous work in differential privacy GAN research and novel loss functions to improve GAN performance. The DP-LkGAN is achieved through a rigorous iterative process involving adjustments to the original GAN architecture and the implementation of differential privacy guarantees. This led to the generation of a new synthetic dataset that shared the same distribution as the original training set. The dataset consisted of handwritten digit images, and the GAN's generated images indicate that with higher levels of noise, the style of each digit is indistinguishable. Further analysis was conducted through parameter tuning in order to find strong values that result in the desired level of privacy. The resulting figures of the DP-LkGAN have shown that this proposed method can generate reasonably private data points._
## Software Implementation
All source code used to generate the results and figures is written in python and executed through [Google Colab](https://colab.research.google.com/) notebooks. The MNIST dataset is built into tensorflow and is directly loaded into the code. 
## Downloading the code 
You can download a copy of all the files in this repository by cloning the git repository:
```
git clone https://github.com/15jjg6/DP-LkGAN.git
```
