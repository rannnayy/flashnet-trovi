#!/bin/bash

source ~/.zshrc
if [ "$(which conda)" != "" ]
then
    echo "Anaconda has already installed."
else
    echo -e "Anaconda has not installed, installing one."
    cd /tmp
    wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
    bash Anaconda3-2022.10-Linux-x86_64.sh -b
    echo "Anaconda installed."
    export PATH="/home/$USER/anaconda3/bin:$PATH"
    source ~/.zshrc
fi
conda init zsh
source ~/.zshrc
condaEnvName="flashnet-trovi-env"
conda create --name $condaEnvName python=3.8 -y
conda activate $condaEnvName
echo "conda activate $condaEnvName" >> ~/.zshrc

which -a pip | grep $condaEnvName
    # get anaconda3's pip for the $condaEnvName
    # Anything installed by this $condaPip will only be available to $condaEnvName

export condaPip=`which -a pip | grep $condaEnvName`
    # export condaPip=/home/daniar/anaconda3/envs/flashnet-trovi-env/bin/pip
$condaPip --version

# Now, install the rest of the dependencies
$condaPip install tensorflow==2.7.0 --no-cache-dir
$condaPip install numpy
$condaPip install pandas
$condaPip install scikit-learn
$condaPip install statsmodels 
$condaPip install matplotlib
$condaPip install future
$condaPip install onnx
$condaPip install mpi
$condaPip install tqdm
$condaPip install pydot
$condaPip install ipympl    # Important to display plot at notebook
$condaPip install seaborn
$condaPip install tabulate
$condaPip install xgboost
$condaPip install catboost
$condaPip install bokeh

# conda deactivate --> To quit the current conda environment
# conda env list   --> To list ALL available conda environment in your machine
# conda remove --name <env_NAME> --all

mkdir -p /tmp/
cd /tmp/
git clone https://github.com/mlperf/logging.git mlperf-logging
$condaPip install -e mlperf-logging
$condaPip install tensorboard
rm -rf mlperf-logging

$condaPip install jupyter
$condaPip install jupyterlab
$condaPip install jupyterthemes
$condaPip install ipykernel

# install ipykernel [on "flashnet-trovi-env"]
$condaPip install ipykernel

# Add conda env to jupyter!!
python -m ipykernel install --user --name=$condaEnvName

# install jupyterthemes
$condaPip install jupyterthemes

# Set Theme
jt -l
jt -t chesterish

conda deactivate