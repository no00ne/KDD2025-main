# CausalMob

This repository contains the Python source code for **CausalMob: Causally Predicting Human Mobility with LLMs-derived Human Intentions toward Public Events**.

## Environments

To run CausalMob, ensure you have the following dependencies installed:

- **Python**: 3.8.0 
- **Pandas**: 1.5.3
- **NumPy**: 1.23.5
- **Torch**: 1.12.1
- **scikit-learn**: 1.3.0

You can install these dependencies using pip:

```sh
pip install pandas==1.5.3 numpy==1.23.5 torch==1.12.1 scikit-learn==1.3.0
```


## Usage

To use CausalMob, download the source code and run the following command in your terminal:

```
python run.py --causal True --input_window 24 --output_window 24 --device cuda:0 --batch_size 24
```

--causal: Set this to True to enable causal predictions.

--input_window: The number of input time steps.

--output_window: The number of output time steps.

--device: Specify the device for computation (e.g., cuda:0 for GPU or cpu for CPU).

--batch_size: Set the batch size for training.

## Data

In our research, we used [Blogwatcher data](https://www.blogwatcher.co.jp/) for human mobility and [Kyodo News data](https://english.kyodonews.net/) for news articles, both data time range is from 2023.04.01 to 2024.03.31.

However due to our data privacy policy, we are unable to share the code. Below is the format of the data used in this repository:

**D**: Day

**T**: Time interval of a day (hour)

**N**: Node of regions

**F**: Features of human mobility (in our research, it is 1, representing human volume)

Data Files:

1. **flows.npy**: Human mobility data with a shape of D × T × N × F. The time interval is 15 minutes, so use --interval = 4 in run.py.
2. **odmetrics_sparse_tensors.pk**: Normalized Origin-Destination (OD) data with a shape of D × T × N × N.
3. **prev_treats_sum.npy**: Public event features in our research area, with a shape of D × T × N × scores (10 scores in our research). This file includes historical event features.
4. **post_treats_sum.npy**: Public event features in our research area, with a shape of D × T × N × scores (10 scores in our research). This file considers only predictable future public events.
5. **poi_distribution.pk**: Normalized Points of Interest (POI) distribution in each region, with a shape of N × POIs (17 categories), we collected POI data from Open Streat Map ([OSM](https://www.openstreetmap.org/)).