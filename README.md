# Channel Fusion Using LSIM in Automatic Sleep Staging

The provided MATLAB codes can be used to replicate the results and generate the figures presented in the following paper.

- S. Karimi and M. B. Shamsollahi, ["A New Post-Processing Method Using Latent Structure Influence Models for Channel Fusion in Automatic Sleep Staging,"](https://ieeexplore.ieee.org/document/9973288) in IEEE Journal of Biomedical and Health Informatics, vol. 27, no. 3, pp. 1569-1578, March 2023, doi: 10.1109/JBHI.2022.3227407.

The codes and data for the papers on Selected Deep Spectrum & Deep Learning Systems have been omitted, and instead, the outputs of these methods are included for easy execution.
I would like to express my gratitude to Huy Phan (pquochuy) for generously providing the codes for xsleepnet and seqsleepnet. Additionally, I would like to thank Hau-Tieng Wu for sharing the codes for the DGDSS method. 


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

![proposed post-processing method](/blockdiag.png)

The output hypnogram of the proposed integrated LSIM fusion of three channels processed by TinySleepNet and the output hypnograms
of selected baseline systems for the Fpz-Cz channel in subject 2.
![The output hypnogram](/hypnogram.png)


## 2. Usage & Dependency.

## Dependency:
     sleep-edf dataset
     [chmm-lsim-matlab-toolbox](https://github.com/sajjadkarimi91/chmm-lsim-matlab-toolbox) 


## Usage:
Before execution, it is necessary to configure the "channel_num" parameter to either 2 or 3 in all subsequent scripts, depending on whether 2-channel or 3-channel fusion is desired.

1. Run [general_lsimfusion_train.m](./general_lsimfusion_train.m) to train 2 or 3 channels LSIM and extract LBSS features.
   
   Note: We uploaded trained LSIM models for fast execution and exact reproduction of results. If you are interested in training from scratch, you can set force_train=1 in Line#11
3. Run [fusion_standard.m](./fusion_standard.m) to perform standard LSIM fusion
4. Run [fusion_integrated.m](./fusion_integrated.m) to perform integrated LSIM fusion with better performances
5. Run [results_summary_fusion.m](./results_summary_fusion.m) to produce results and metrics for proposed Post-Processing method.
6. Run [results_summary_singlechannel.m](./results_summary_singlechannel.m) to produce results and metrics for original deep sleep staging systems.


