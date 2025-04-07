# SiameseDuo++

Data stream mining, also known as stream learning, is a growing area which deals with learning from high-speed arriving data. Its relevance has surged recently due to its wide range of applicability, such as, critical infrastructure monitoring, social media analysis, and recommender systems. The design of stream learning methods faces significant research challenges; from the nonstationary nature of the data (referred to as concept drift) and the fact that data streams are typically not annotated with the ground truth, to the requirement that such methods should process large amounts of data in real-time with limited memory. This work proposes the SiameseDuo++ method, which uses active learning to automatically select instances for a human expert to label according to a budget. Specifically, it incrementally trains two siamese neural networks which operate in synergy, augmented by generated examples. Both the proposed active learning strategy and augmentation operate in the latent space. SiameseDuo++ addresses the aforementioned challenges by operating with limited memory and limited labelling budget. Simulation experiments show that the proposed method outperforms strong baselines and state-of-the-art methods in terms of learning speed and/or performance. To promote open science we publicly release our code and datasets.

# Paper

You can get a free copy of the accepted version from Zenodo ([link](https://zenodo.org/records/15127904)) or arXiv (TBA).

For the published version, visit the the publisher’s website ([link](https://www.sciencedirect.com/science/article/pii/S0925231225007556)).

# Instructions
Please check the “instructions.txt” file.

# Requirements
Please check the “requirements.txt” file.

# Citation request
If you have found our paper and/or part of our code and/or datasets useful, please cite our work as follows:

Kleanthis Malialis, Stylianos Filippou, Christos G. Panayiotou, Marios M. Polycarpou, SiameseDuo++: Active learning from data streams with dual augmented siamese networks, Neurocomputing, Volume 637, 2025, 130083, ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2025.130083.