# Transformer Capsule Neural Network for Twitter Sentiment Analysis

Jacob Fitzgerald and Kyler Nelson 2023

## Project Description
The objective of this project is to attempt to create a highly accurate TSA (Twitter Sentiment Analysis) model by combining the strong representational abilities of a transformer with the powerful feature recognition present in a capsule neural network. In this case, Sentiment extraction is treated as a binary classification problem, with sentiment either being positive or negative. The models architecture is as follows:


## Results
**Capsule Network Only Training Accuracies**
![](/docs/notransacc.png)

**Transformer and Capsule Network Training Accuracies**
![](/docs/transacc.png)

## Running the Code

There are quite a few prerequisets that need to be satisfied before the code can actually be run. Make sure to complete every step under the requirments section before running the code.

The datasets are very large, making data parsing and training very CPU intensive processes. For this reason, preprocessed data is saved in the project directory. If you ever get stuck waiting on a  "CLEANING INPUT" message, this means that this preprocessed data can't be found, and data processing will take about an hour. To run the pretrained models, navigate to the project directory (the one containing this file), and run one of the following commands to test a singular dataset:

**Only Capsule:**
```
python transcapsnet.py --testing --weights ./model/onlycaps/trained_model.h5 --dataset sentiment140 --embedding_dataset sentiment140 --trans_off
```

**Transformer and Capsule**
```
python transcapsnet.py --testing --weights ./model/transcaps/trained_model.h5 --dataset sentiment140 --embedding_dataset sentiment140 
```

The '--dataset' argument can be set to any of the following:
* sentiment140
* airlines
* stocks
* sanders

**DO NOT** change the --embedding_dataset argument, as that refers to the dataset used to train the model, which the embeddings are based off of. 

If you'd like to train a model, be aware that it took us ~1 hour per epoch (training on a NVIDIA 2070 Super), mainly due to the large dataset size. To train the model run the following commands. Like before, you can choose from any of the avaiable datasets.

**Only Capsule:**
```
python transcapsnet.py --dataset sentiment140 
```

**Transformer and Capsule:**
```
python transcapsnet.py --dataset sentiment140 --trans_off
```

You should theoretically be able to switch out the dataset for any of the others listed above, but this code hasn't be run on anything other than sentiment140 for training, so they're not gaurenteed to work.

After training, you can find the saved model weights in the './results' subdirectory. If you would like to save them, copy them to a new folder in the './model' subdirectory. **RUNNING THE MODEL TRAINING WILL OVERWRITE ANYTHING IN THE './results' SUBDIRECTORY.** So, if you don't want your trained model to be deleted, make sure to save it somewhere else.


# Requirments

This code relies on quite a few python libraries, a few of which have unusual setups. The most important thing to remember is, if you want to train on a GPU, you have to have a tensorflow version **LESS THAN OR EQUAL TO 2.10.0**. Additionally, this code has not been tested with any package versions beyond the ones listed in this section, so there are no garuentees it will work with newer packages.

**Python Packages**

You can install all python packages either manually with pip by running:
```
pip install package_name=version
```

Or all at once by navigating to the project directory and running:
```
pip install -r requirments.txt
```

*Packages Required*
```
numpy
tensorflow=2.10.0
keras=2.10.0
nltk
PIL
pandas
matplotlib
```

**Additional Setup**

Additionally, after installing the NLTK python package, it is **NECESSARRY** to open up a repl and run:
```
import nltk
nltk.download("all")
```

This will download all the language processing libraries to your **USER DIRECTORY**. If you'd like them somewhere else, you can run:
```
import nltk
nltk.download()
```

And it will open a GUI which allows you to choose the download location. Make sure to select the "all" option. 

It should be noted that there have been reports of NLTK not working properly if the libraries are not installed directly to the default location in the user directory. If you do install them somewhere else, it's recomended to install them into the empty "./corpus" subdirectory in this project, as the code **SHOULD** be able to detect them there.



## Data Organization
This project should include all the data needed to run. However, in the case that it's not possible to download the entire project, you'll have to manually download the datasets to the proper subdirectories.

The project expects the datasets to be formulated as follows:
- Datsets will be located in the "./data" subdirectory
- Each dataset will be located in it's own folder named datasetname
- All data will be in a singular CSV file within this folder named datasetname.csv
	- The program will create an additional file, data_clean.csv in the folder, which contains processed data
- There needs to be two empty folders within this directory as well, named train and test
	- The program will place training and testing data artifacts in these folders in order to speed up future runs
	- The program will **NOT** create these folders on it's own, the user must do so

## Data Processing
Data is processed in different unique ways depending on the dataset. The general flow is:
1. Data is read in as a pandas dataframe
2. Text data is tokenized, stemmed, and lemmetized
4. Data is split 80/20 into train/test
3. A vocabullary is constructed based on tokens
4. An embedding matrix is constructed for the whole vocabulary
5. Text is turned into sequence data using only the top 50,000 words
6. Data is padded/truncated to a fixed sequence length
7. Label data is transformed into categorical data

The preprocessor will attempt to save data at intermediate steps to help speed up execution.

## Datasets
[Sentiment140](http://help.sentiment140.com/for-students/)
[Sanders Analytics](https://github.com/zfz/twitter_corpus)
[Airlines](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
[Stocks](https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset)

## Acknowledgements

Preprocessing Code: https://github.com/rohanrao619/Twitter_Sentiment_Analysis
Capsule Network Code: https://github.com/XifengGuo/CapsNet-Keras
Transformer Code: https://machinelearningmastery.com/the-transformer-model/

