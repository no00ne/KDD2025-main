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