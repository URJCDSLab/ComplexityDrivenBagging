# ComplexityDrivenBagging

This repository contains the supplementary results of the Complexity-driven Bagging methodology.
This algorithm is a version of the Bagging algorithm that considers training samples with different degrees of complexity. The basic case contemplates easy, classic and hard samples that are obtained by sampling with weights $w_e$, $w_u$ and $w_h$, receptively. Easy (hard) samples are those where the easy (hard) instances have a higher probability of being chosen. Classic samples are those where every point has the same probability of being picked as in the Bagging algorithm. The algorithm holds two parameters: s to obtain intermediate weights between the already defined ones so as to obtain samples of intermediate complexity and $\alpha$ to give more emphasis to the easiest and hardest cases. 

Alternatives studied have included:
* Easy. Only obtaining training samples with $w_e$, that is, giving more weight to easy cases.
* Hard. Only obtaining training samples with $w_h$,  that is, giving more weight to difficult cases.
* Combo. Sampling with $w_e$ and $w_h$.
* Combo split s. Sampling with $w_e$ and $w_h$ and with (s-1) intermediate weights between them.

The repository is organized as follows:
* The folder “Datasets” contains both the real and the artificial datasets utilized for the experiments.
* The folder “Results” collects the different results obtained in the experiments. There are results concerning the alternatives studied but also more disaggregated results of the final algorithm.
