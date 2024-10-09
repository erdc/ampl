# An Automated Machine Learning Pipeline (AMPL)

* Why are we looking at NNs and ML models
* A set of steps are defined for the process of training a NN and ML models
* Name the steps and mention breifly what those steps are
* Describe how the results are displayed with the percentage of points under a 20,10,5,2.5,0.5
* With the steps defined, improving the workflow to allow for both non-technical and
  technical users.
* After improvements were made, the next goal was to allow for automating the process of training a network.

Machine Learning techniques have risen in popularity as ML has shown to be useful in providing an expert level response to predicting values, recognizing patterns, and identifying objects in images. While working through applying ML to ["SYNTHETIC CFD ESTIMATION FOR BLACKHAWK AIRFOIL DRAG COEFFICIENT"](https://doi.org/10.2514/6.2024-1230), and ["ESTIMATING KINETIC ENERGY REDUCTION FOR TERMINAL BALLISTICS"](https://link.springer.com/article/10.1007/s00521-023-09382-3), it was noted that the steps for applying ML was similiar enough to where a single workflow could be designed to handle both of these problems along with previously unexplorered problem spaces. The steps that were taken for both the rotorcraft and ballistics problem were Feature Importance, Hyperparamter optimization which is searching for superior model parameters, training the best models returned from hyperparamter optimization, and evaluating the performance of the best models. This documentation will describe the details of each of the previously mentioned steps. Details will also be provided on how to utilize each of the steps individually or as an automated workflow. In this documentation we will describe an automated machine learning pipeline and present results of applying the pipeline to sample problems.

The workflow steps used to design the models used for predicting values for the Ballistics and Rotorcraft work were the same steps. Since the Ballistics work and the rotorcraft work use very different data, but the steps in the workflow were the same, a general workflow that could design ML models for many different problems was desired. Having a general method would reduce effort in the beginning stages of working on a new problem or dataset and allow for exploration of methods and techniques to create better models. The general method would also remove the need to implement each step from ground up and would improve the timeline from receiving the data to having a reasonably performing model. The method for the general workflow is called "An Automated Machine Learning Pipeline" and the method will fit the following criteria:

* Easy to get started for novice machine learning users
* Have sufficient options for expert users
* The tools used in each step must be able to be modifiable or replaceable to allow improvements to the pipeline
* Able to run each step in the workflow as a solo component
* Eliminate the need to manually design ML model to allow more time to be focused on adding new or enhancing existing Steps in the automated machine learning pipeline

## Steps to run AMPL

1. [Anaconda (Conda) setup](#anaconda-conda-setup)
2. [Install AMPL](#install-ampl)
   * [GitLab setup](./README_git.md?ref_type=heads#gitlab-setup)
      * [Option 1: Get SSH key](./README_git.md?ref_type=heads#option-1-get-ssh-key-recommended)
      * [Option 2: Get HTTPS Access Token](./README_git.md?ref_type=heads#option-2-get-https-access-token-needed-to-clone-the-repo-as-well-as-pullpush)
   * [Clone AMPL repository](#clone-ampl-repository)
   * [Install AMPL in your conda environment](#install-ampl)
      * [Initial Steps for Running AMPL](#initial-steps-for-running-ampl)
3. [AMPL setup and Configuration file](#ampl-setup-and-configuration-file)
    * [Suggested directory structure for organizing AMPL](#suggested-directory-structure-for-organizing-AMPL)
4. [Two Methods for Interfering with AMPL](#Two-Methods-for-Interfering-with-AMPL---API,-CLI)
    * [AMPL - API](#ampl-api)
    * [AMPL - CLI](#ampl-cli)

## Anaconda (Conda) setup

Please download and install Anaconda python if it has not previously been installed. Installatiion instructions can be found within [README_anaconda](./README_anaconda.md?ref_type=heads#install-anaconda) if needed by the user.

### Anaconda Setup

It is important to maintain an up-to-date version of Anaconda. Even if a user already has Anaconda, please follow the steps for Updating Conda and Anaconda. Not having an updated version of conda is the most commonly experienced error by new users.

1. Update Conda and Anaconda (Recommended for all users)

Update the conda package manager to the latest version in your base environment
```shell
conda update -n base conda -y
```

Use conda to update Anaconda to the latest version in your base environment
```shell
conda update -n base anaconda -y
```

```shell
conda activate    
```
You should then see (base).

2. Create a new conda environment named `ampl` and install the libraries and modules required for AMPL.

Note: If the user experiences an error during these installs, the most common source for that error is the need to update conda and anaconda as seen in the previous step.

```shell
conda create -n ampl python=3.11 pandas numpy yaml scikit-learn jupyter recommonmark scikit-learn-intelex plotly::plotly anaconda::sphinx -y
```

```shell
conda install -n ampl -c conda-forge optuna matplotlib imbalanced-learn sphinx-gallery cloud_sptheme -y
```

```shell
conda activate ampl
```

```shell
python -m pip install --upgrade pip myst-parser joblib
```

```shell
pip install shap pillow requests xgboost jinja2 more_itertools optuna-integration tensorflow
```

## Setting up gitlab

The user will need to setup their SSH key to git in order to pull the source code for AMPL. This step can be skipped by users who have already setup their git SSH key. This process for setting up the git SSH key will be the same whether a user is running locally or on HPC as a user will need to store an SSH key for each machine they use to access git. Instructions for setting up git are found here within the [README_git] (./README_git.md?ref_type=heads#GitLab-setup)

### Creating AMPL Code and Run/working Directory - for both API and CLI users

The directory structure for working with AMPL is the same regardless of using the API or CLI method to interface with AMPL. This recommeneded directory structure will help with organization and determining where to put the AMPL repository, as well as provide a convienient way to organize the directory structure so that ML models and information (plots, statistics, etc..) are easily accessed by the user. A user can use any directory structure they would like, but will have to customize their yaml input file to account for any differences between their chosen directory structure and the recommended directory structure.

#### Recommended directory structure

The user may use multiple methods for creating a directory structure, however, the example below assumes the user is working from commandline. For the example working with AMPL, we will use an open-source concrete dataset. When a user is working with their own data they should rename any references to 'concrete' with their own dataset name.

1. Create a parent folder for AMPL Code

ex: $ mkdir <ampl_dir> 
```shell
mkdir AMPL
```

2. Move into the <ampl_dir>

ex: cd <ampl_dir> 
```shell
cd AMPL 
```

3. Create AMPL repository code directory:

whatever you would like to name your designated AMPL repository code directory 
ex: mkdir <ampl_code_dir>
```shell
mkdir code
```

4. Create directory for AMPL runs:

whatever you would like to name your directory to hold all of the AMPL runs.
ex: mkdir <all_run_dir> 
```shell
mkdir all_run_dir
```

5. Move into the <all_run_dir>

ex: cd <all_run_dir> 
```shell
cd all_run_dir
```

6. Create directory for a specific run of AMPL:

whatever you would like to name your designated AMPL working directory based on the data being used.
here we create concrete_run_dir for holding information for the concrete data example. Please name
concrete to something different based on the data you are working with.
ex: mkdir <dataset#_run_dir>

```shell
mkdir concrete_run_dir
```

Note: Repeat step six to create the structure to maintain the data for different data sets and runs. Example directory structure is provided below.

#### Example Directory Structure

The following is an example directory structure of what AMPL will look like when the code is downloaded and a user is working with multiple datasets. This example shows where the ampl code is located, and provides two examples of where the run data is stored. The first example the user has a csv file and uses one yaml file for specifying how AMPL will run. The second example the user has a sqlite data file and two different yaml files for specifying how AMPL with run when working with the same dataset. AMPL will accept both csv and sqlite data, however, the user will need to modify the yaml file based on which type of data is being used. Examples for working with csv or sqlite are provided below.

Note: When working with AMPL, the normal use case is to create a Folder named 'AMPL' to store everything. This folder is the root folder and should be the folder a user opens when working with AMPL whether through the [API](#ampl-api) or through the [command-line option](ampl-cli).

```shell
├── <ampl_dir>
│   ├── <ampl_code_dir>
│   │   ├── <ampl>
│   │   │   ├──<docs>
│   │   │   ├──<examples>
│   │   │   ├──<src>
│   │   │   ├──<tests> 
│   ├── <all_run_dir>
│   │   ├── <dataset1_run_dir>
│   │   │   ├── <data_dir>
│   │   │   │   ├── <data1.csv>
│   │   │   ├── <config_file.yml>
│   │   ├── <dataset2_run_dir>
│   │   │   ├── <data_dir>
│   │   │   │   ├── <data2.sqlite>
│   │   │   ├── <config_file1.yml>
│   │   │   ├── <config_file2.yml>
```

### Clone AMPL repository

1. Navigate to your AMPL repository code directory <ampl_code_dir> created in the previous step [Recommended directory structure](#recommended-directory-structure):

ex: cd <ampl_dir>/<ampl_code_dir>
```shell
cd code
```


The following is an example of the user's directory structure 
```shell
cd ../code
  ├── ampl_dir
->│   ├── code
  │   ├── all_run_dir
```


4. Go to the AMPL Git repo page in your browser, select Code and then there will be two options to `Clone`. Copy the URL based on whether you are using the SSH Key or HTTPS option (SSH is recommended). Use the following command to clone the repo into your AMPL working directory:


example using ssh:
ex: git clone <copied URL>
```shell
git clone git@github.com:erdc/ampl.git
```

Note: If you don't have access or are getting a permission error from Git, please refer to [Option 1: Get SSH key](./README_git.md?ref_type=heads#option-1-get-ssh-key-recommended)

#### Install AMPL in your conda environment

 Note for Windows users: Please open Anaconda power shell as an administrator for the commands to work.

1. Activate the conda environement you created previously while installing Anaconda.

  ex: conda activate <env_name>
  ```shell
  conda activate ampl 
  ```

2. Install the needed packages.

  ```shell
  # Navigate to the inside the code folder. 
    ├── AMPL
  ->│   ├── code
  ```
  
  ex: pip install -e <ampl_code_dir>
  ```shell
  pip install -e ampl
  ```

3. If the install was successful then you will see soemthing similiar to:

  ```shell
     Successfully built ampl
     Installing collected packages: ampl
     Successfully installed ampl-0.0.3
  ```

# Test AMPL in your env by running the following:


1: Make sure you are in the right conda environment, i.e. the ampl environment
  ex: conda activate <env_name>
  ```shell
  conda activate ampl
  ```
2: Run some tests that are part of the ampl code by first navigating to the tests directory
  ex: cd <ampl_code_dir>/tests

  ```shell
  cd ampl/tests
  ```
 
3: Test the pipeline connections
  ```shell
  python -m unittest test_pipeline_nn
  ```

  The command above that tests the pipeline connection does a small test to confirm if the connections are set up properly.  It starts with using Optuna to find the best trial and then runs 10 epochs to train that best trial.  After it is done running, it displays the reuslts of the test in a table that includes the layer type, output shape, and number of parameters. Below the table are more detials about the parameters that are used in the test run.

### AMPL setup and Configuration file

#### Initial Steps for Running AMPL

AMPL utilizes configuration file(s) to setup a study/run.

When AMPL is executed, a .yaml file must be passed in as an argument for running in either CLI or API mode. The .yaml file contains a path to the location of the input data, which is used for training, validation, testing, as well as the path to the results directory. The location of these directories are relative to the aforementioned path provided in the .yaml in addition to many other configuration settings.

[Default Config File](./src/ampl/default_config.yml)

#### Suggested directory structure for organizing AMPL

You will want to create a directory for your specific AMPL run to keep your runs organized, especially if you will be running AMPL on multiple datasets. This is where all your project related folders should be created and where your ampl_config.yml should reside. We recommend creating a separate folder to store data within it and refer to it in the config file.

1. Navigate to your AMPL running directory.  The run directory is typically located in the home directory <ampl_run_dir>, previously refered to as AMPL/all_run_dir in examples:


  ex: cd <ampl_dir>/<ampl_run_dir>
  from the AMPL directory
  ```shell
  cd all_run_dir
  ```

  If following along from the example the path will be
  cd ../../../all_run_dir
  ```shell
  The folder structure will look like this
    ├── ampl_dir
    │   ├── code
  ->│   ├── all_run_dir
  ```
  
2. Create a directory for your run and navigate to this directory.  Use a name that indentifies the dataset you will be using. Since the example that we will be using is based on a concrete dataset, we will name the directory appropriately:


  ex: mkdir <dataset_run_dir>
  ```shell
  mkdir concrete_run_dir
  ```
  ```shell
  The folder structure will look like this
    ├── ampl_dir
    │   ├── code
    │   ├── all_run_dir
  ->│   │   ├── concrete_run_dir
  ```
