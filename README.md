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


## Results




