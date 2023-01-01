# Automated-EEG-Sleep-Staging

A MATLAB toolbox for EEG-based Sleep Stage Classification from preprocessing, feature extraction, feature selection, dimension reduction, and classification using SVM and KNN.

Codes and data for the following paper are extended to different methods:

A New Post-Processing Method using Latent Structure Influence Models for Channel Fusion in Automatic Sleep Staging


## 1. Introduction.

This package includes the prototype MATLAB codes for Automated EEG Sleep Staging.

The implemented methodes include: 

  1. Various feature extraction methods, including 
     * Multiscale permutation entropy


  2. Several dimension reduction methods including PCA, LDA and TSNE
  3. Multiple classifiers SVM, KNN, NeuralNets



## 2. Usage & Dependency.

## Dependency:
     sleep-edf dataset
     https://github.com/sajjadkarimi91/SLDR-supervised-linear-dimensionality-reduction-toolbox


## Usage:
Run "main_run.m" or "main_binary.m" to analyze the sleep staging.
