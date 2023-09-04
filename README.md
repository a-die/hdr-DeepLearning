# Automatic Recognition of Hand-Drawn Chemical Molecular Structures Based on Deep Learning

Chemical molecular structural formula can express chemical knowledge more directly and conveniently, so it plays an important role in academic communication. **Hand-drawing chemical molecular structural formulas** is a routine task for students and researchers in the field of chemistry. If hand-drawn chemical molecular structural formulas can be converted into machine-readable data, computers can process and analyze these chemical molecular structural formulas, so **Greatly improve the efficiency of chemical research**.

The main research methods of this study are as follows:

1. In terms of **Dataset**, several chemical molecular structural formulas are drawn by hand. In addition, this article uses the modified source code *RDKit* to convert *SMILES* codes into chemical molecular structural formula images, using random bond lengths, random rotation and other operations , to simulate the hand-painted chemical molecular structure image, and then combine image enhancement, degradation and other methods to further process the image to obtain a large number of synthetic images. Also, *img to img* task was done using *diffusion*.
2. In terms of **model**, use the *PyTorch* deep learning framework to build the model of the encoder and decoder architecture, and use variant networks such as convolutional neural networks as encoders to convert the features of hand-drawn chemical molecular structure images into fixed The encoded state of the shape is then fed into the decoder to obtain a machine-readable encoding.

## Requirements

The dependencies required for the *hdr* environment of this project

```
conda env create -f environment.yaml
conda activate hdr
```

## Script

The experiment done by *pipline_stages* is a comparative test of synthetic image method

The experiment done by *size_test* is a comparison test of the number of synthetic images

The experiment done by *SD_HD* is a synthetic image: real hand-painted image scale experiment

# Dataset

If you need to obtain the complete dataset, please contact the corresponding author by email:weiliu@hnucm.edu.cn