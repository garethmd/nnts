# Evaluating the effectiveness of predicting covariates in LSTM Networks for Time Series Forecasting

This repository is the official implementation of [Evaluating the effectiveness of predicting covariates in LSTM Networks for Time Series Forecasting](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements
Ensure you have the following installed:
- Python 3.x

Clone the repository or copy the source code into a directory on your machine and cd into the root folder of the code

Run venv to create a virtual environment
``bash
python -m venv venv
```

Activate the virtual environment
``bash
source venv/bin/activate
```

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

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




## Evaluation


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 