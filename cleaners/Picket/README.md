# Picket: Guarding Against Corrupted Data in Tabular Data during Learning and Inference
Picket is a system that safeguards against data corruptions during both training and deployment of machine learning models over tabular data. Details of the system can be found in our paper: https://arxiv.org/abs/2006.04730. Please cite our paper if you use the code.

## Environment Requirements
We run the experiments with the following dependencies:
* pytorch v1.0.1
* matplotlib v3.2.1
* pandas v0.23.4
* scikit-learn v0.22.1
* gensim v3.8.1
* tqdm v4.45.0
* adversarial-robustness-toolbox v1.2.0

For poisoning attack at training time, we use the implementation by the Github repo: **Poisoning-Attacks-with-Back-gradient-Optimization**

## Set up with a Conda Virtual Environment
* First, download Anaconda (not miniconda) from https://www.anaconda.com/products/individual. Follow the steps for your OS and framework.
* Second, create a conda environment (Python 3.6+). For example, to create a Python 3.7 conda environment, run:
```shell
conda create -n picketEnv python=3.7
```
* Upon starting/restarting your terminal session, you will need to activate your conda environment by running
```shell
conda activate picketEnv
```
**The following two steps should be run in the activated virtual environment:**
* Run `bash install.sh gpu` for dependency installation within a conda3 environment if a gpu is available, otherwise run `bash install.sh cpu`. The default CUDA version is 10.0. Modify intall.sh accordingly if you use a gpu with other CUDA versions.
* Run `python -m ipykernel install --user --name=picketEnv` to add your virtual environment to Jupyter

**In a new terminal outside the virtual environment:**
* Run `jupyter notebook` to Launch the Jupyter Notebook App.
* Open one of the notebooks, change the kernel to picketEnv and run it. The following screen shot illustrates how to change the kernel:
![Alt text](./changeKernel.png?raw=true "Change the kernel to picketEnv")

## Demos
The following two notebooks provide demos showing the usage of Picket.
* **notebooks/Demo-Titanic.ipynb**: The notebook demonstrates how to integrate our system into a machine learning pipeline. The noise is random.
* **notebooks/Demo-Wine.ipynb**: Same as the previous one except that the noise is adversarial.

## Experiments
The following notebooks run the experiments with the same settings in the paper. The datasets can be found at https://drive.google.com/file/d/1onXCORjgTGizlL_QjZQ2LoVgJ0uDE5Q_/view?usp=sharing.
* **notebooks/Experiments-Main.ipynb**: The notebook runs experiments that evaluate the end-to-end performance of Picket.
* **notebooks/Experiments-MicroBenchmark-TwoStream.ipynb**: The notebook validates the effectiveness of the two stream design of PicketNet.
* **notebooks/Experiments-MicroBenchmark-EarlyFiltering.ipynb**: The notebook validates the effectiveness of early filtering.
* **notebooks/Experiments-MicroBenchmark-VictimSampleDetector.ipynb**: The notebook validates the effectiveness of one victim sample detector per class.
* **notebooks/Experiments-MicroBenchmark-ArtificialNoiseType.ipynb**: The notebook validates the effectiveness of the artificial noise injected for detector training.
* **notebooks/Experiments-MicroBenchmark-Structure.ipynb**: The notebook validates the ability of Picket to capture the structure of the data.
* **matlabCode/downstreamEffectNN.m**: The script evaluates the effect of filtering on the downstream neural network after notebook/Experiments-Main.ipynb.

Deactivate the virtual environment using `conda deactivate`, `source deactivate`, or `deactivate` depending on the version, when it's no longer need.