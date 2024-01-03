# A New Post-Processing Method Using Latent Structure Influence Models for Channel Fusion in Automatic Sleep Staging

MATLAB codes for reproducing the paper results and figures.

The codes and data for the papers on Selected Deep Spectrum & Deep Learning Systems have been omitted, and instead, the outputs of these methods are included for easy execution.

## 1. Introduction.

This package includes the MATLAB codes for Automated EEG Sleep Staging using LSIMs.

Sleep stage dynamics can be adequately represented
using Markov chain models to improve classification accuracy. The
present study proposes a new post-processing method based on
channel fusion using Latent Structure Influence Models (LSIMs).
The proposed method develops and examines two channel-fusion
algorithms: the standard LSIM fusion and the integrated LSIM
fusion, in which the latter is more efficient and performs better.
The proposed LSIM-based method simultaneously incorporates
the nonlinear interactions between channels and the sleep stage
dynamics. In the first step, existing sleep staging systems process
every data channel independently and produce stage score
sequences for each channel. These single-channel scores are then
projected into belief space using the marginal one-slice parameter
of all channels by LSIM fusion algorithms. The logarithms
of marginal one-slice parameters are concatenated to obtain logscale
belief state space (LBSS) features in the standard LSIM
fusion. In the integrated LSIM fusion, integrated LBSS (ILBSS)
features are formed by combining the LBSS features of several
LSIMs. By utilizing four recently developed sleep staging systems,
the proposed method is applied to the publicly available SleepEDF-
20 database that contains five AASM sleep stages (N1, N2, N3,
REM, and W). Compared to single-channel (Fpz-Cz, Pz-Oz, and
EOG) results, integrated LSIM fusion results have a statistically
significant improvement of 1.5% in 2-channel fusion (Fpz-Cz and
Pz-Oz) and 2.5% in 3-channel fusion (Fpz-Cz, Pz-Oz, and EOG).
With an overall accuracy of 87.3% for 3-channel post-processing,
the integrated LSIM fusion system offers one of the highest overall
accuracy rates among existing studies.

The structure of the proposed post-processing method for 2-channel PSG with the standard LSIM fusion (red blocks use train labels): 

![proposed post-processing method](/blockdiag.jpg)

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
