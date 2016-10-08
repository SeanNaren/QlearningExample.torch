# TorchQLearning
![TorchPlaysCatch](https://github.com/SeanNaren/TorchQLearningExample/raw/master/images/torchplayscatch.gif)

Torch plays catch! Based on Eder Santanas' [implementation](https://gist.github.com/EderSantana/c7222daa328f0e885093) in keras. Highly recommend reading his informative and easy to follow blog post [here](https://edersantana.github.io/articles/keras_rl/).

Agent has to catch the fruit before it falls to the ground. Agent wins if he succeeds to catch the fruit, loses if he fails.

## Dependencies

To install torch7 follow the guide <a href="http://torch.ch/docs/getting-started.html">here</a>.

Other dependencies can be installed via luarocks:

```
luarocks install optim
luarocks install image
```

## How to run

To train a model, run the `Train.lua` script. You can configure parameters such as below:

```
th Train.lua -epoch 1000 #Configures the number of epochs. More parameters are available, check scrip.t
```

## Visualization

To visualise the agent playing the game after training a model, use the `TorchPlaysCatch.lua` script using qlua rather than th as below:

```
qlua TorchPlaysCatch.lua
```

Much like the train script, there are configurable options. Check the script for more details!

## Acknowledgements
Eder Santana, Keras plays catch, (2016), GitHub repository, https://gist.github.com/EderSantana/c7222daa328f0e885093
