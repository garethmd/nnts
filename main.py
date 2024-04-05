from dataclasses import dataclass

import pandas as pd

import src.data.datasets as datasets
import src.data.tsf


@dataclass
class Metadata:
    """Class for storing dataset metadata"""
    path:str
    dataset:str
    context_length: int
    prediction_length: int
    freq: str
    seasonality:int

metadata = Metadata(
    path = '/Users/garethdavies/Downloads/hospital_dataset.tsf',
    dataset = 'hospital',
    context_length = 15,
    prediction_length = 12,
    freq = 'M',
    seasonality = 12
)

def main() -> None:
    # Load CSV file into a Pandas DataFrame
    datai = src.data.tsf.convert_tsf_to_dataframe(metadata.path)
    df=pd.DataFrame(datai[0])
    data = pd.concat([datasets.unpack(df.iloc[x], freq=metadata.freq) for x in range(len(df))])

    
    # Construct a time series dataset
    dataset = datasets.TimeseriesDataset(data, metadata.context_length, metadata.prediction_length)

    # Print the dataset
    print(dataset)



if __name__ == "__main__":
    main()