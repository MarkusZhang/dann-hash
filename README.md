# dann-hash
Implementation of Domain Adversarial Network for hashing
(refer to the paper: [unsupervised domain adaptation by backpropagation](https://arxiv.org/pdf/1409.7495.pdf))

To run the main training process, run run.py (you will need to have [ml_toolkit](https://github.com/MarkusZhang/ml_toolkit) installed to run the program). It will run both training and testing, and save the models and results.
The main training process uses pairwise similarity loss for both source and target data.

The folder "expr_suites" contains the scripts for running DANN with some other settings, refer to the README under the folder
