This repository contains a multilabel text classifier for the Reuters news data. The documents have been classified into TOPICS, which are used as the labels for the training. 

You can retrain the model in one of two ways. The first is to open the jupyter notebook `Modeling_nn.ipynb`. Because the data is included in the data folder, the notebook should run without adjustments (but please feel free to tell me if I'm wrong!).

The second is to run the `train.py` file. Similar to the notebook, it collects all the functions and data locally and trains the model. It also generates a figure that looks at the training and validation accuracy over the epochs in the training.

The model is a neural network, which is compiled in `classifier.py`. One possible way to improve the model is to adjust the layers of the network (there are currently 3 fairly large layers). Additionally, although the accuracy as the optimization metric is sufficient, trying more meaningful accuracy metrics might give lift to the model. 

I decided to use a neural network, because it was an interesting challenge. I toyed around with a few other algorithms and given more time I would like to explore how it stands up in training time, accuracy, and prediction latency. 

If there are questions or concerned, don't hesitate to contact me. 
