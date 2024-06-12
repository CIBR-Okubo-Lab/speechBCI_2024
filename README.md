# Brain-to-Text Benchmark '24

This repository contains code and model checkpoints of the second-place solution for the [Brain-to-Text Benchmark '24](https://eval.ai/web/challenges/challenge-page/2099/overview) submitted by TeamCyber (Yue Chen and Xin Zheng in [Okubo Lab](https://cibr.ac.cn/science/team/detail/975?language=en) at the [Chinese Institute for Brain Research, Beijing (CIBR)](https://cibr.ac.cn/)).

Please note that our implementation is not optimized for speed (for example, use of bidirectional GRU or the model ensembling).

This competition is based on the data described in [Willett et al. (Nature, 2023)](https://www.nature.com/articles/s41586-023-06377-x) and our repository heavily relies on the following code prepared by the organizers at Neural Prosthetics Translational Laboratory at Stanford University.

- [TensorFlow implementation](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder) ('TF' in the chart below)
- [PyTorch implementation](https://github.com/cffan/neural_seq_decoder) ('PT' in the chart below)

We would like to thank the participant of the study and the organizers for making this precious data and the code public.

# Requirements
We used [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) to test our models.  
`environment.yml` included in this repository can be used to recreate our conda enviromnent.

To run `eval_competition.py` and `notebooks/LLM_ensemble.ipynb`: 
1. Pull [LanguageModelDecoder](https://github.com/fwillett/speechBCI/tree/main/LanguageModelDecoder) into the project directory.
2. Pull [rnnEval.py](https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/utils/rnnEval.py) and [lmDecoderUtils.py](https://github.com/fwillett/speechBCI/blob/main/NeuralDecoder/neuralDecoder/utils/lmDecoderUtils.py) into `<path to project>/NeuralDecoder/neuralDecoder/utils/`. 
3. Follow the [instructions](https://github.com/fwillett/speechBCI/tree/main/LanguageModelDecoder) to build and run the language model decoder.

# Model training
To train the model, run `neural_decoder_train.py`. 

In our final submission, we trained two different models (Model 1 and Model 2 in the chart below) and ensembled them (see the last section on ensembling). The configuration for these two models can be found in `conf/config_1.yaml` and `conf/config_2.yaml`. You can specify which configuration to use in `neural_decoder_train.py` line 141.

Due to non-deterministic behavior of cuDNN, the training code might not produce exactly the  same results as our submission. Therefore, we have also included checkpoints for these two models.


## Hyperparameters

We include a hyperparameter comparison chart for convenience.

### RNNDecoder

| RNN Parameter | Description | TF | PT | Ours |
| --- | --- | --- | --- | --- |
| nUnits | Number of units in each GRU layer | 512 | 1024 | 1024 |
| nLayers | Number of GRU layers | 5   | 5   | 5   |
| bidirectional | whether to use bidirectional GRU or not | False | True   | True   |
| Kernel Size | Number of input feature time bins stacked together as a single input for the RNN | 14  | 32  | 32  |
| Stride | Describes how many time bins the RNN skips forward every step | 4   | 4   | 4   |
| L2  | L2 regularization cost | 1e-5 | 1e-5 | 1e-5 |
| Dropout | Probability of dropout during training | 0.4 | 0.4 | 0.4 |
| WhiteNoiseSD | Standard deviation of white noise added to input data for regularization | 1.0 | 0.8 | 0.8 |
| constantOffsetSD | Standard deviation of constant offset noise added to input data to improve robustness against non-stationary feature means | 0.2 | 0.2 | 0.2 |
| Batch Size | Number of sentences included in each mini-batch | 64  | 64  | 128 |

<br>  

### Original optimizer
- [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [linear learning rate decay](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html)

| Parameter | Description | Value |
| --- | --- | --- |
| Learning Rate | Linearly decaying learning rate | 0.02 to 0.0 |
| β1  | coefficients used for computing running averages of gradient | 0.9 |
| β2  | coefficients used for computing running averages of the square of gradients | 0.999 |
| ε   | term added to the denominator to improve numerical stability | 0.1 |

<br>

### Our optimizer
- [SGD with momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- [Step learning rate decay](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)

| Parameter | Description | Model_1 | Model_2 |
| --- | --- | --- | --- |
| lr | base learning rate | 0.1 | 0.1 |
| momentum | momentum for SGD | 0.9 | 0.9 |
| nesterov | whether to use Nesterov momentum or not | False | True |
| step_size | how many steps to wait before lr decay | 4000 | 5000 |
| gamma | multiplicative factor for the lr decay | 0.1 | 0.1 |

<br>

# Model ensembling

To obtain the final prediction, we performed ensembling of two models. More specifically, model 1 and model 2 each predicted a sentence given neural activity, and [Llama2 7b chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) (LLM optimized for dialogues) was used to pick the sentence that had higher score.  
Please see `notebooks/LLM_ensemble.ipynb` for the code and example outputs.
