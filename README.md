# Adaptive Scheduled Multi-Task Learning

This is an implementation of our paper:

P. ZareMoodi, G. Haffari, ”Adaptively Scheduled Multitask Learning: The Case of Low-Resource Neural Machine Translation,” Proceedings of the 3rd Workshop on Neural Machine Translation and Generation, co-located with EMNLP 2019.

## Requirement
This code requires my modified version of PyTorch to run on GPUs. I have modified the PyTorch source code as the official version has not supported some of the
backpropagation-through-backpropagation operations on GPU at the time of implementation. You need to clone the following repository and install it from source. The process may take up to 4-5 hours, and I highly recommend installing an Anaconda environment.
https://github.com/pooryapzm/pytorch

## Using the code

An example can be found in the example folder.

```
cd example
bash train_script
```

## References

This code is built upon [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

## TODO
Add a detailed explanation of the code and parameters.
