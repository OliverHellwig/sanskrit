# Overview

This directory contains data and code for the following paper:

```
@inproceedings{hellwig-2021-chr,
	author = {Hellwig, Oliver and Sellmer, Sven and Nehrdich, Sebastian},
	title = {Obtaining More Expressive Corpus Distributions for standardized Ancient Languages},
	booktitle = {Proceedings of the CHR 2021},
	year = {2021}
}
```

# Workflow

## Building the data

The data are extracted from the CLTK library Latin corpus and lemmatized using the Collatinus software.
This repository provides the routines for working with these lemmatized data.

1. Download the five files containing the lemmatized texts from [Google drive](https://drive.google.com/drive/folders/1hwhUT33dx7vX-3PHj5gbC80zg0rPM4Te?usp=sharing), unzip into data/
2. Run code/R/build-data-latin-collatinus.R (Collatinus > internal format).
3. Run code/R/build-data.R (for c++).

## Running the sampler

Compile the cpp files in code/cpp in a Visual Studio project. Run the project.

## Evaluation

As an example of how to evaluate the results of the sampler, we provide the R script for inspecting the structures of word reuse, 
Sec. 5.2 of our paper, in code/R/evaluate-word-reuse.R
