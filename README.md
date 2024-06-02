# Markowitz Random Forest

## Project installation
### Create a new venv environment and activate it.
python3 -m venv mrf-env
source mf-env/bin/activate
### Install the neccessary requirements.
pip install --upgrade pip
pip install -r requirements.txt 



## Project Structure
- The folder tree_weighting includes the implementations of the considered tree weighting methods in object-oriented style
- data_loaders.py is the interface for loading the 15 datasets used in the experiments
- Some datasets are loaded directly using the ucimlrepo package and the other are retrieved from the data folder.
- Experiment and analysis files exist for every learning task
- The results are stored in the corresponding results folders
