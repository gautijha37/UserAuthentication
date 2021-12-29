# Self Organizing Maps based Clustering for Keystroke Dynamics
### MIES Group 2 Term Project

This repository contains the extraction of relevant features from the data and process of keystroke dynamics by performing a dimensionality reduction using Kohonen Self Organising Maps a.k.a SOMs and inturn, do the unsupervised classification and authentication of users

Libraries Used:
1. numpy
2. matplotlib
3. standard Python Libraries

# Steps for Data Acquisition:

>def extract_keylog_features(path) 

_[keystroke_feature_extractor.py](keystroke_feature_extractor.py)_

This functons takes the keylog features with the timestamp in two cases ie... hold time for a particular key and transition time between 2 keys pressed. 

Note: This functions also eliminates the cases of outliers where the time stored for the key pressed is >1 sec.

The data for the different individuals is then, loaded in the Jupyter Notebook and further processing is done.

# Steps for dimesionatility reduction using SOM:

The training of an SOM is governed by hyperparameters learning rate decay time τη, and for every epoch t, of training they are updated to learning rate η and neighbourhood size σ. Implemented in _[SOM.py](SOM.py)_

They are set using the following coefficients:

`initial learning rate, η0 = 1;  initial neighbourhood size, σ0 = 8;  initial neighbourhood size, σ0 = 500;  neighbourhood size decay rate, τσ = 650;  n = 15`

> def winning_neuron(self, x, cox)

This function takes the input vector(_x_) and find the Euclidian distance. It further, updates the clustering centres(_cox_) ie... weights closest to the input vector is determined.

> def neighbourhood_weights(self, p_win, q_win, sigma)

This function do the comparison of the Euclidian distance for each of the weight parameters from the winning centres(_p_win, q_win_) and decides a neighbourhood region around the winning region and weights would be updated significantly only in that region. 

> def update_weights(self, x, cox, eta, Topo_neigh)

This function update the weights and return Topo_neigh ie... perceptron with weights closest to the input vector is determined. As per the neighbourhood weights obtained, weight parameters are obtained by significant amounts only in the neighbouring regions. The weight connecting neuron (p, q) and input node is updated.

> def set_of_winning_neurons(self, x_vectors, cox_vectors)

This function tells the final set of winning neurons for each input

Then, we used clustering techniques **K-means** to cluster the data.

# Execution: 

1. Run [main.py](main.py)

Note: All the .py files should be in the same directory

# References:

1. [Term Project Report](report.pdf), MIES Group 2, Autimn 2021, IIT Kharagpur.
2. Teuvo Kohonen. Self-organizing maps: ophmization approaches. In Artificial neural networks, pages 981–990. Elsevier, 1991.
3. Teuvo Kohonen. Exploration of very large databases by self-organizing maps. In Proceedings of international conference on neural networks (icnn’97), volume 1, pages PL1–PL6. IEEE, 1997.

