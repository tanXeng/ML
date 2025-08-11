# Transformer model from scratch

An encoder-decoder model built with only PyTorch (aside from the tokenizer). This project is a close replica of the original transformer from [here](https://arxiv.org/pdf/1706.03762). Apart from LayerNorm coming before the attention heads instead of after, and a few other small changes, this project stays mostly loyal to the architecture in the paper.

This model is trained to translate English sentences to German. It was trained on the WMT 2014 German ↔ English, of approximately 4.5 million english-german pairs.

## Project Structure

```bash
Transformer_from_scratch/
├── Translation_model/
│ ├── init.py # module init
│ ├── config.py # model training configuration
│ ├── layers.py # transformer layers
│ ├── model.py # full transformer model definition
│ └── README # this README file
├── train.py # model training script
├── translator.py # inference script to translate sentences
├── translator.pth # saved trained model weights
└── utils.py # empty
```

## How to train

1. Change dataset within the train.py file to your desired dataset.
2. Change the config.py file as desired.
3. Run `python path/to/Transformer_from_scratch/Translation_model/train.py`

## How to run inference

1. Change the inference config within the translator.py file (e.g. temperature)
2. Run `python "Your english sentence here"`
