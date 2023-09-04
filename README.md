---
datasets:
- agender
- mozillacommonvoice
- timit
- voxceleb2
inference: true
tags:
- speech
- audio
- wav2vec2
- audio-classification
- age-recognition
- gender-recognition
license: cc-by-nc-sa-4.0
---

# Model for Age and Gender Recognition based on Wav2vec 2.0 (24 layers)

The model expects a raw audio signal as input and outputs predictions 
for age in a range of approximately 0...1 (0...100 years) 
and gender expressing the probababilty for being child, female, or male. 
In addition, it also provides the pooled states of the last transformer layer. 
The model was created by fine-tuning [
Wav2Vec2-Large-Robust](https://huggingface.co/facebook/wav2vec2-large-robust) 
on [aGender](https://paperswithcode.com/dataset/agender), 
[Mozilla Common Voice](https://commonvoice.mozilla.org/), 
[Timit](https://catalog.ldc.upenn.edu/LDC93s1) and 
[Voxceleb 2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html).
For this version of the model we trained all 24 transformer layers.
An [ONNX](https://onnx.ai/") export of the model is available from 
[doi:10.5281/zenodo.7761387](https://doi.org/10.5281/zenodo.7761387). 
Further details are given in the associated [paper](https://arxiv.org/abs/2306.16962) 
and [tutorial](https://github.com/audeering/w2v2-age-gender-how-to).

# Usage

```python
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = self.gender(hidden_states)

        return hidden_states, logits_age, logits_gender



# load model from hub
device = 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name)

# dummy signal
sampling_rate = 16000
signal = np.zeros((1, sampling_rate), dtype=np.float32)


def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y


print(process_func(signal, sampling_rate))
#    Age       child      female      male
# [[ 0.3079211 -1.6096017 -2.1094327  3.1461434]]

print(process_func(signal, sampling_rate, embeddings=True))
# Pooled hidden states of last transformer layer
# [[-0.00752167  0.0065819  -0.00746342 ...  0.00663632  0.00848748
#    0.00599211]]
```
