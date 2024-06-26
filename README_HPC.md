# Running AMPL on Carpenter

The following guide is for running AMPL on the HPC Carpenter. Users must have access for carpenter for this guide to be helpful. However, users may examine this file for a general guideline of what steps they need to take to get AMPL on a HPC machine in which they have access. The general steps to running AMPL on Additional HPC machines will be the same.


## General steps to running AMPL on HPC

The following steps will be described in detail in the following sections

1. Load module cse/anaconda3/latest
2. Create Conda env
3. SSH key
4. Git lab add pub key
5. git clone
6. pip install -e .
7. Install all AMPL required libraries

## Logging on to HPC
The following steps will explain how to connect to HPC Carpenter for new users:
If you have an HPCMP account and have never connected to an HPC system before, follow the instructions at https://centers.hpc.mil/users/index.html for your operating system to:
1.	Install the HPCMP Kerberos Client Kit,
2.	Obtain a ticket using your CAC, and
3.	Login to an HPCMP system

NOTE: The path to Carpenter (carpenter.erdc.hpc.mil)

## Running on the HPC
The following guide will explain how to run codes on HPC Carpenter for new users:
[HPC Carpenter Quick Start Guide](https://centers.hpc.mil/users/docs/erdc/carpenterQuickStartGuide.html)


## Installing Anaconda 
You will need access to Anaconda on the HPC system to create a conda environment. The conda environment will be used to install the packages needed to run AMPL. 

You can use preinstalled Anaconda3 modules on HPC.

NOTE: These steps are specific to Carpenter

```shell 
module load gcc/12.2.0
```

```shell
module load cseinit
```

```shell
module load cse/anaconda3/latest
```

```shell
bash
```

```shell
conda init bash
``` 

Now, either restart the putty or the terminal to activate the newly initiated conda: 

```shell
source .bashrc
```


## Creating the Anaconda Environment in HPC to Use AMPL - with GPU support 

Follow the following instructions to create your conda environment and install the packages needed to use AMPL:

```shell
# Create the conda environemnt 
conda create -n ampl python=3.11 pandas numpy yaml jupyter recommonmark -y
```

```shell
# install more things to the ampl conda environment
conda install -n ampl -c conda-forge scikit-learn scikit-learn-intelex plotly sphinx optuna matplotlib imbalanced-learn sphinx-gallery cloud_sptheme -y
```

```shell
# activate the environment
conda activate ampl
```

```shell
# use pip to install a few things
pip install --upgrade pip myst-parser
```

```shell
# install a few more things with pip
pip install shap pillow requests xgboost jinja2 more_itertools optuna-integration 
```

```shell
# and a few more installs
pip install tensorflow[and-cuda] nvidia-cudnn-cu12 tensorrt --extra-index-url https://pypi.nvidia.com
```

## Creating Working Directory

You want to run your jobs in your work directory ($WORKDIR) on the HPC system, so you need to create a directory designated for AMPL. 

1.	Navigate to your work directory:
```shell
cd $WORKDIR
```

### IMPORTANT! DO NOT SKIP SETTING UP THE DIRECTORY STRUCTURE!!!!

Please follow the following guide for setting up the AMPL directory structure in the README. The setps in the section will walk the user through creating the propor directory structure along with downloading the sample concrete dataset and setting up the .yml file.


  !!!!!! DO NOT MISS THIS LINK TO SET UP THE DIRECTORY STRUCTURE AND FOR CREATING THE YAML FILE !!!!!!
[Suggested directory structure for organizing AMPL](./README.md?ref_type=heads#Suggested-directory-structure-for-organizing-AMPL)
  !!!!!! DO NOT MISS THIS LINK TO SET UP THE DIRECTORY STRUCTURE AND FOR CREATING THE YAML FILE !!!!!!



The proper AMPL directory structure should now look like the following if all steps in the "Create the AMPL directory structure" were followed. Do note that the user may use different names for the folders and files in this directory structure, but will have to rename file paths accordingly so it is not recommended until after gaining experience in working with AMPL.

```shell
  # the current folder structure should look like this 
    ├── AMPL
    │   ├── code
    │   ├── all_run_dir
    │   │   ├── concrete_run_dir
    │   │   │   ├── concrete_data
    │   │   │   │   ├── concrete.csv
  ->│   │   │   ├── ampl_config.yml
```

### Super Important to Know!!!

Many HPC systems purge files in your work directory ($WORKDIR) that have not be touched for 30 days and there is no way to recovery these files. You will want to regularly back up files from your work directory to avoid losing your work.

## Create the SSH Key and Add the Key to Git

```shell
Note: There are two methods to access git: HTTPS and SSH.
HTTPS prompts you to enter userid and access token information everytime you push/pull, but with SSH you dont have to remember or enter any access token.
```

If you need to add an SSH key for your machine to be able to pull from git, use the following guide:
[README_git-create-the-ssh-key](./README_git.md?ref_type=heads#create-the-ssh-key)

After you have created a git key or if you have created a git key on your machine in the past, you must add the git ssh key to git. Instructions for adding the git key can be found at [README_git-Add-SSH-Key-to-GitLab](./README_git.md?ref_type=heads#Add-SSH-Key-to-GitLab)

## Cloning the AMPL Git Repo

You will need to clone the AMPL git repo into your AMPL working directory to access the AMPL codes. In order to access the repo from the HPC, you will need to make sure that you have provided Git your SSH key ([instructions found in the previous section](Create-the-SSH-Key-and-Add-the-Key-to-Git)).

The following link will take you to the HPC git login page, but will need a git account from the HPC help desk which can be reached with the following email address: ```servicenow@helpdesk.hpc.mil```.
[HPC Git Login](https://gitlab.hpc.mil/users/sign_in)

Go to the AMPL HPC Git repo page in your browser located at the following link: [gitlab.hpc.mil](https://gitlab.hpc.mil/AMPL/ampl).
Next, select Code and then there will be two options to `Clone`. Copy the URL based on the SSH Key or HTTPS option that you are using. The following command to clone the repo into your AMPL working directory at the following location:

```shell
  # the current folder structure should look like this 
    ├── AMPL
  ->│   ├── code
    │   ├── all_run_dir
    │   │   ├── concrete_run_dir
    │   │   │   ├── concrete_data
    │   │   │   │   ├── concrete.csv
    │   │   │   ├── ampl_config.yml
```

```shell
# example using ssh:
# ex: git clone <copied URL>

git clone git@gitlab.hpc.mil:AMPL/ampl.git
```

## Installing AMPL

After downloading the AMPL repository, the next step is to install AMPL to a conda environment. Navigate to the code folder within the AMPL folder structure located here:

```shell
  # the current folder structure should look like this 
    ├── AMPL
  ->│   ├── code
    │   │   ├── ampl
    │   ├── all_run_dir
    │   │   ├── concrete_run_dir
    │   │   │   ├── concrete_data
    │   │   │   │   ├── concrete.csv
    │   │   │   ├── ampl_config.yml
```

Install the neededed packages with the following command

```shell
# ex: pip install -e <ampl_code_dir>
pip install -e ampl
```

## PBS Script

The following steps are necessary to create a PBS script to allow the user to request resources and run AMPL on Carpenter. To help the user, a sample PBS script may be found within the examples folder under the name 'ml-pipeline.pbs'. A user will need to edit some details within this file to get things to function. Before attempting to run AMPL using a PBS script, it is required that the user has followed the steps in sections: [Installing Anaconda](Installing-Anaconda) and [Creating the Anaconda Environment in HPC to Use AMPL - with GPU support](Creating-the-Anaconda-Environment-in-HPC-to-Use-AMPL---with-GPU-support)

The full pbs script is provided here for example and will be exaplained in the following sections:

```shell
#!/bin/bash
#PBS -A <project_ID>
#PBS -q standard
#PBS -l select=1:ncpus=128:mpiprocs=128:ngpus=1
#PBS -l walltime=<number of hours for the job to run. exp: "24:00:00">
#PBS -N <name of the job>
#PBS -j oe
#PBS -l application=python
#PBS -m be
#PBS -M <Your Email Address>

module load gcc/12.2.0
module load cseinit
module load cse/anaconda3/latest
module load cuda

source /app/CSE/CSE.20240128/Release/anaconda3-2023.03-1/etc/profile.d/conda.sh
conda activate ampl

cd /p/work/<userID>/AMPL/all_run_dir/concrete_run_dir/concrete_data/
python -m ampl ampl_config.yml
```



### Suggested: Folder location for PBS script

Set up your PBS script in your work directory ($WORKDIR) on the HPC system.

1.	Navigate to your work directory, create a directory for your PBS scripts, and navigate to this directory:

```shell
cd $WORKDIR
```

```shell
mkdir pbs_scripts
```

```shell
cd pbs_scripts
```

### PBS Script Preamble

The following lines in 'pbs_scripts/ml-pipeline.pbs' represent the preamble of the pbs script. You will need to fill in your <project_ID>, choose a <run_name> to identify your job, and enter your <email_address> to be notified when your job begins and ends execution.

To get your <project_ID> in HPC system, use the following command and use the Subproject field as your <project_ID>

```shell
show_usage
```
![show_usage](docs\images\show_usage.png)

PBS Script

```shell
#!/bin/bash
#PBS -A <project_ID>
#PBS -q standard
#PBS -l select=1:ncpus=44:mpiprocs=44 
#PBS -l walltime=24:00:00
#PBS -N <run_name>
#PBS -j oe
#PBS -l application=python
#PBS -m be
#PBS -M <email_address>
```

#### Use the following to request a CPU node or GPU node

To request a CPU node use the followin line in the PBS script

```shell
#PBS -l select=1:ncpus=44:mpiprocs=44
```

Alternatively, use the following line as a replacement in the PBS script above to access a GPU node:

```shell
#PBS -l select=2:ncpus=22:mpiprocs=22:ngpus=1 
```

### PBS Script Load libraries

After the PBS preamble section, the next section of the PBS script will load the libraries that AMPL will need in order to run. This section will not require any edits by the user.

```shell
module load gcc/12.2.0
module load cseinit
module load cse/anaconda3/latest
module load cuda
```

### PBS Script ready conda

The next piece of the PBS script will activate the conda environment that will be used to run the AMPL code. Enter your HPC <user_name> into the source file path to locate your conda environment and insert the <env_name> of the environment you would like to activate:

```shell
source /p/home/<user_name>/anaconda3/etc/profile.d/conda.sh # path to your conda.sh on HPC
conda activate ampl
```

### Navigate to YAML File Location and Run AMPL

The following lines in "pbs_scripts/ml-pipeline.pbs" are used to point to the directory where your codes are located and to execute the AMPL code. The line ```python -m ampl ampl_config.yml``` in the PBS script will run all steps in AMPL in succession. Further commandline parameters for running AMPL through the CLI-interface method may be found in the [Getting Started CLI](./README.md?ref_type=heads#Getting-started-CLI) section of the README file:

```shell
cd /p/work/<userID>/AMPL/all_run_dir/concrete_run_dir/

python -m ampl ampl_config.yml
```


Your pbs script should look similar to the following for Carpenter, for other HPC systems update the proper cores/nodes (ex: 192) in the PBS script below

```shell
#!/bin/bash
#PBS -A <project_ID>
#PBS -q standard
#PBS -l select=1:ncpus=128:mpiprocs=128:ngpus=1
#PBS -l walltime=<number of hours for the job to run. exp: "24:00:00">
#PBS -N <name of the job>
#PBS -j oe
#PBS -l application=python
#PBS -m be
#PBS -M <Your Email Address>

module load gcc/12.2.0
module load cseinit
module load cse/anaconda3/latest
module load cuda

source /app/CSE/CSE.20240128/Release/anaconda3-2023.03-1/etc/profile.d/conda.sh
conda activate ampl

cd /p/work/<userID>/AMPL/all_run_dir/concrete_run_dir
python -m ampl ampl_config.yml

```

### Execute AMPL on HPC-Carpenter

Once you are done editing the pbs script, use the following command in the command line on the HPC to launch your job to run the pipeline code. This is the last step to being able to run AMPL on HPC-Carpenter:

```shell
qsub ml-pipeline.pbs
```

After executing the line above, a job will start on HPC. You should receive an email as since as your job starts to execute

#### Optional method to launch HPC node in interactive mode
NOTE: This step is specific to certain HPC, where you are able to edit your .bashrc file. Some HPC systems ask you NOT to edit the .bashrc file and will have instructions for editing some type of .personal_bashrc file.

Launching AMPL using interactive mode can be helpful to debugging and provides a way for the user to view what is happening as it happens. The only issue with this method is that it may not be able to reconnect to the node if for some reason the user's internet goes out while the job is running.


Edit .bashrc to add

```shell
vi ~/.bashrc
```

Add the following lines to your .bashrc file

```shell
export CUDNN_PATH="$HOME/.conda/envs/ampl/lib/python3.9/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64"
export PATH="$PATH":"/usr/local/cuda/bin"
```

Use the following command to activate these changes:

```shell
source ~/.bashrc
```

#### GPU
```shell
module load cuda # for gpu support
```

After create and editing your PBS script the following line from commandline will request a GPU Node in interactive mode

```shell
qsub -I -A <Project_ID> -l select=1:ncpus=128:mpiprocs=128:ngpus=1 -l walltime=2:00:00 -q standard
```
