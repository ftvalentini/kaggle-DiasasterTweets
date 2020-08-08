# Disaster Tweets - NLP Kaggle Dataset

## Introduction
The goal of this competition is to identify tweets that are related to natural disasters or emergencies such as fires, floods, earthquakes, etc. The data is provided in this [LINK](https://www.kaggle.com/c/nlp-getting-started) and is composed by a train dataset with 7613 tweets and a test dataset with 3263 tweets. This is a supervised problem since there are labels for all the records.  
The data has a simple structure. ID, keyword (if available), location (if available), tweets (280 chars) and label.

## Preparation
We did some exploratory analysis to get a sense of the data we had at our hands. Some highlights:  
* Balanced classes
* Location is missing on half of the dataset
* More frequest Keywords are related to disasters

We also cleaned the text with usual string preprocessing techniques such as removing badly formatted characters and stopwords, we also trimmed the text and  removed some twitter specifics like mentions and hashtags.  
In order to make it more model friendly we replaced many social media abbreviations by their full length format. Ex: afaik -> "as far as I know".

## Models

First of all  we ran a Naive Bayes model and a Logistic Regression to quickly have some kind of benchmark without incurring into more complex models. We got an F1 score of around 0.67 in test for both models.

Nowadays, the models that are shining in NLP tasks are the neural networks with their different architectures.  
We moved to a Google Colab environment to be able to run such resource intensive models. We used TensorFlow to fit the neural networks.

At this point we spent considerable time researching and reading about the different approachs and their differences. We started an interative process where we tried them and compared results after doing hyperparameter tuning.  
Among the different architectures we tried there were: Forward NN, LSTM, GRU and BIGRU but also  pretrained models such as Glove.   
In the end, what gave us the best results (not only in our test set but also in the submission to Kaggle) was a BERT based model leveraging over Google's efforts. It's not surprising since BERT is (or was back then) prettyy much the state of the art technique. 

## Results

We achieved roughly an F1 score of 0.83 in the submission (that is not seen at during training) which is quite remarkable comparing to our benchmarks. It was also quite good in comparison to other participants as we moved up the ladder signifacntly.  
The model was able to classify with acceptable confidence if a non seen tweet during training was refering to some natural disaster or not, which is useful as an extra tool to quickly spot dangerous situations to the population or to later on get better understanding of how the turmoil was seen by the people present there.



