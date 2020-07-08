# NFCalib

An official code for the [NFCalib paper](https://www.youtube.com/watch?v=dQw4w9WgXcQ)


### Data

We are using a number of common tabular datasets as well as well known CIFAR and MNIST.
To download it all just run: `cd src/data && python3 get_data.py`


### Usage

First, one should build a conda environment:

`conda env create -f environment.yml`

Currently there are three ways to run or recreate experiments:

1. Running a single experimnet from a jupyter notebook: `notebooks/NFCalib_notebook.ipynb`
2. Running a single experiment from a python script: `src/nfcalib.py`
3. Running a grid of experiments on a cluster: `cluster/run_cluster_jobs.py`


### Contacts

In case of any problems ping me: tg@meretemev or mrartemev.me@gmail.com