# Library for generating and testing recommendation models based on recurrent neural networks (LSTM)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The library was created in the frame of a diploma thesis at the Faculty of Information Technology.

The algorithm for creating models for generating recommendations was designed within the practical part of the diploma thesis. 
In addition to the algorithm for creating and training modules, 
there is also a library for testing success (offline evaluation). 
The library lists collaborative filtering algorithms (user-knn, item-knn, MF) as well as the simple 
popularity and reminder model to compare the success of models.

## Link to this thesis

Martı́nek, Ladislav. Doporučovacı́ modely založené na rekurentnı́ch neuronových sı́tı́ch. Diplomová práce. 
Praha: České vysoké učenı́ technické v Praze,
Fakulta informačnı́ch technologiı́, 2020.


## Abstract

This diploma thesis deals with matters of recommendation systems. 
The aim is to use recurrent neural networks (LSTM, GRU) to predict the subsequent interactions 
using sequential data from user behavior. Matrix factorization adapted for datasets with 
implicit feedback is used to create a representation of items (embeddings). 
An algorithm for creating recurrent models using the embeddings is designed and implemented in this thesis. 
Furthermore, an evaluation method respecting the sequential nature of the data is proposed. 
This evaluation method uses recall and catalog coverage metrics. 
Experiments are performed systematically to determine the dependencies on the observed methods and hyperparameters. 
The measurements were performed on three datasets. 
On the most extensive dataset, I managed to achieve more than double recall against other recommendation techniques, 
which were represented by collaborative filtering, reminder model, and popularity model. 
The findings, possible improvement by hyper-parametrization, 
and different possible means of model improvement are discussed at the end of the work.

## Requirements
* Python version 3.6 and higher
* Python packages which are specified in requirements.txt

## Limitations
Algorithms and models created in the thesis are used for research and evaluation of the behavior of these models. 
Therefore, they are not optimized for working with extensive data. 
This is only a basic implementation of algorithms by definition.

## Input data format
The input data file is in CSV format, and it is a quartet `[user_id, item_id, timestamp, interaction_weight]`.

Other data files are CSV files with user_ids (train, valid, test), 
embeddings file (item_id, latent factors split by ','), similar items, or LSTM train checkpoint, 
which are generated with models or script in this repository.

## Configuration
The configuration takes place through the configuration file listed in `config/experiment.yml`, 
where there are also comments on individual points. 
Further details are given in the chapter Implementation in the thesis itself and Appendix C.

## Run
Examples of running individual modules are given in jupyter notebooks.

* Data splitter - `src/splitAndSaveData.sh`
* MF - `src/mf.sh`
* LSTM training - `src/trainLSTM.sh`
* evaluation - `src/evaluation.sh` (
There are many evaluation options that are configured using a file and can be used to evoke embedding similarities.)


