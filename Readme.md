# CausalMob

This is the Python source codes for CausalMob: Causally Predicting Human Mobility with LLMs-derived Human Intentions toward Public Events.

## Enviroments
python: 3.8.0 

pandas: 1.5.3

numpy: 1.23.5

torch: 1.12.1

sklearn: 1.3.0


## Usage

Download the source code and run the following code in the terminal

```
python run.py --causal True --input_window 24 --output_window 24 --device cuda:0 --batch_size 24
```