# RFold: Towards Simple yet Effective RNA Secondary Structure Prediction

![GitHub stars](https://img.shields.io/github/stars/A4Bio/RFold)  ![GitHub forks](https://img.shields.io/github/forks/A4Bio/RFold?color=green) <!-- ![visitors](https://visitor-badge.glitch.me/badge?page_id=A4Bio.RFold) -->

## Introduction

The secondary structure of ribonucleic acid (RNA) is more stable and accessible in the cell than its tertiary structure, making it essential in functional prediction. Though deep learning has shown promising results in this field, current methods suffer from either the post-processing step with a poor generalization or the pre-processing step with high complexity. In this work, we present RFold, a simple yet effective RNA secondary structure prediction in an end-to-end manner. RFold introduces novel Row-Col Softmax and Row-Col Argmax functions to replace the complicated post-processing step while the output is guaranteed to be valid. Moreover, RFold adopts attention maps as informative representations instead of designing hand-crafted features in the pre-processing step. Extensive experiments demonstrate that RFold achieves competitive performance and about eight times faster inference efficiency than the state-of-the-art method.

## Model overview

We show the overall RFold framework.

<p align="center">
  <img src='./assets/overview.png' width="600">
</p>

## Benchmarking

We comprehensively evaluate different results on the RNAStralign, ArchiveII datasets.

<p align="center">
  <img src='./assets/rnastralign.png' width="300">
</p>

<p align="center">
  <img src='./assets/archiveii.png' width="300">
</p>

## Colab demo

We provide a Colab demo for reproducing the results and testing RNA sequences by yourself:

<a href="https://colab.research.google.com/drive/1rAWP7evVLc7cbIP3KzPr5ZlHTVo9A57g?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- [[Colab]](https://colab.research.google.com/drive/1rAWP7evVLc7cbIP3KzPr5ZlHTVo9A57g?usp=sharing) -->

## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
TBD
```

## Feedback
If you have any issue about this work, please feel free to contact me by email: 
* Cheng Tan: tancheng@westlake.edu.cn