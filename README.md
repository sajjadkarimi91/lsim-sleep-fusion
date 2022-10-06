# Automated-EEG-Sleep-Staging

A MATLAB toolbox for EEG-based Sleep Stage Classification from preprocessing, feature extraction, feature selection, dimension reduction, and classification using SVM and KNN.

Codes and data for the following paper are extended to different methods:

Diffuse to fuse EEG spectra–intrinsic geometry of sleep dynamics for classification.


## 1. Introduction.

This package includes the prototype MATLAB codes for Automated EEG Sleep Staging.

The implemented methodes include: 

  1. Various feature extraction methods, including 
     * Multiscale permutation entropy
     * Statistical features
     * AR coefficients
     * Spectrul entropy
     * Hjorth parameters mobility and complexity
     * Approximate entropy
     * Lyapunov exponent
     * Correlation dimension
     * Mel-frequency cepstral coefficients


  2. Several dimension reduction methods including PCA, LDA and TSNE
  3. Multiple classifiers SVM, KNN, NeuralNets


     


## 2. Usage & Dependency.

## Dependency:
     sleep-edf dataset
     https://github.com/sajjadkarimi91/SLDR-supervised-linear-dimensionality-reduction-toolbox
     Kijoon Lee (2022). Fast Approximate Entropy (https://www.mathworks.com/matlabcentral/fileexchange/32427-fast-approximate-entropy), MATLAB Central File Exchange.
     Valentina Unakafova (2022). Permutation entropy (fast algorithm) (https://www.mathworks.com/matlabcentral/fileexchange/44161-permutation-entropy-fast-algorithm)
     

## Usage:
Run "main_run.m" or "main_binary.m" to analyze the sleep staging.
