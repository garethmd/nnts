# Evaluating the effectiveness of predicting covariates in LSTM Networks for Time Series Forecasting

This repository is the official implementation of Evaluating the effectiveness of predicting covariates in LSTM Networks for Time Series Forecasting

## Requirements and Setup
Ensure you have the following installed:
- Python 3.x

Fork, clone or download the source code from the reppository into a directory on your machine and navigate to the projects | rnn-covariates directory 
in a terminal window.

Run the following commmand to create a virutal environment, install the required packages and download the datasets.

```bash
make all
```

Activate the virtual environment
```bash
source covenv/bin/activate
```

Load the datasets
```bash
make sync_data_from_drive
```



## Training
To train the model(s) in the paper, run this command:
```bash
python train.py all data all
```


Alternatively to train a specific model for a dataset, run this command
```bash
python train.py dataset_name data_path [model_name] [results_path] [generate_metrics]
```
Use the following options:

- dataset_name
    - hospital
    - tourism
    - traffic
    - electricity

- data_path
    - data/hospital_dataset.tsf
    - data/tourism_monthly_dataset.tsf
    - data/traffic_weekly_dataset.tsf
    - data/electricity_hourly_dataset.tsf

- models
    - base-lstm
    - seg-lstm

- results_path (eg results)

- generate_metrics
    - true
    - false

This will create a folder where the results of each model will be stored along with the state_dict of the pytorch model weights created during training. 
Each execution of train creates a set of 24 models covering a range of scenarios:
 - 5 univariate models with different seed values
 - 6 models with 1 covariate at various correlation values
 - 6 models with 2 covariates at various correlation values
 - 6 models with 3 covariates at various correlations values 
 - 1 model with 2 covariates (skip x_1)  

Each model will have a corresponding json file that contains the details of the scenario along with test error metrics including MAE, RMSE, and sMAPE. Once training has completed for all models a results.csv file will be generated which collates the json files for each scenario together to allow for further analysis.   

If you choose to generate metrics a selection of plots used in the paper as well as serialised model predictions and targets are saved into the chosen results directory. 

## Notebooks 
3 Jupyter notebooks are available for which also perform training of the same scenarios, but allow for greater control and customisation. 
 - 01-base-lstm.ipynb - Trains datasets using the base-lstm model
 - 02-seg-lstm.ipynb - Trains datasets using the seg-lstme model
 - 04-generate-results.ipynb - Generates the plots and serialises predictions. 


## Naming Convention
The artifacts that are created in the results directory follow a specific naming convention:

scenario results: cov-{k}-pearsn-{PCC}-pl-{H}-[skip-1]-seed-{s}.json
scenario test target values: cov-{k}-pearsn-{PCC}-pl-{H}-[skip-1]-seed-{s}_y.pt
scenario test prediction values: cov-{k}-pearsn-{PCC}-pl-{H}-[skip-1]-seed-{s}_y_hat.pt
scenario metrics across forecast horizon trajectories: cov-{k}-pearsn-{PCC}-pl-{H}-[skip-1]-seed-{s}_metrics.pt

Where:  
k - Is the number of covariates (1,2 or 3) 
PCC - Is the correlation coefficient (0.5  - 1.0)  
H - Is the forecast horizon (prediction length)  
skip-1 - Indicates that there are two covariates at t=1 and t=3. (ie we omit the covariate that would be at t=2)  
s - Is the seed number.  



## Licence
See [LICENSE](LICENSE) file for details