
This repository contains code that was used on the paper "A Bimodal Learning Approach to Assist Multi-sensory Effects Synchronization" accepted at IJCNN 2018.

The code is divided into two parts:
- Audioset dataset download
-- Used to download audioset data from youtube and divide into audio and video
- Bimodal neural network architecture and experiments
-- Neural network that uses both audio and video as inputs
-- Experiments that show the Bimodal architecture performed better in our cases



## Audioset dataset download

[get_dataset_from_youtube.py](get_dataset_from_youtube.py)
Directy download from youtube the videos and audio files of youtube audioset.
Inside the script you will note the following hardcoded files : 2000max_subset_unbalanced_train_segments.csv, subset_eval_segments.csv,etc... This csv files are derived from the audioset [found here](https://research.google.com/audioset/download.html)

usage: 
```
python get_dataset_from_youtube.py --train --eval
```

[data/extract.sh](segments/extract.sh)
Used to extract subsets of the full audioset dataset. Used to alleviate the unbalance between classes.
This script reads how many features you want to extract from the audioset and the class names of the audioset labels that you want to extract

In our project tried to limit to a max of 2000 examples per label (thus 2000max in the csv). 

##### TODO
- Remove hardcored values and files

## Bimodal neural network architecture and experiments

Use Keras + Tensorflow to create a neural network to predict the labels associated with the audio and video samples combined.

The Bimodal architecture is presented on [Multimodal.ipynb](Multimodal.ipynb)
We also ran experiments on audio and video only networks to check if the bimodal network as really an improvement. These experiments can be found on the other jupyter notebooks (such as [2000video.ipynb](2000video.ipynb)). Also we ran a lot of visualization to check the learning process of the bimodal network, the visualizations can be found on [KERAS-VIS_activation_maximization.ipynb](KERAS-VIS_activation_maximization.ipynb)

You can view the class model activations for the video network in [https://www.youtube.com/watch?v=dTVbsootmiA]()

##### TODO
- Cleanup legacy code
- Improve file structure

