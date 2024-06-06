# Play it Straight: An Intelligent Data Pruning Technique for Green-AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and resources for the research paper "Play it Straight: An Intelligent Data Pruning Technique for Green-AI". Our proposed "Play It Straight" algorithm aims to reduce the computational and environmental costs of training AI models by strategically pruning the training dataset without compromising performance.

## Key Features

* **Green-AI Focus:** Employs intelligent data pruning to minimize the environmental footprint of AI model training;
* **Play It Straight Algorithm:** Introduces a novel algorithm combining active learning (AL) and repeated random sampling for effective dataset reduction;
* **Comparative Implementations:** Provides scripts for training models with the whole dataset, a pure AL approach, repeated random sampling, and the proposed "Play It Straight" algorithm;
* **Reproducibility:** Facilitates replication of our research results.

## Repository Structure

* **`train_whole_dataset/`:** Contains a Python script (`train_model.py`) for training a model using the entire dataset (baseline approach).
* **`play_it_straight/`:**
    * `main_al.py`: Trains a model using a pure active learning strategy;
    * `main_rs2.py`: Trains a model using the repeated random sampling algorithm;
    * `main_play_it_straight.py`: Implements the proposed "Play It Straight" algorithm for model training.
