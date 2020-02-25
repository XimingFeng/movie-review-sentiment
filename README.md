# Movie Review Sentiment Classification

## Description

This project is to build a REST API that take movie review as input and output sentiment classification of the review.

## Technology stack

- Language preprocessing: Gensim
- Model: Keras with Tensorflow backend
- API: Flask

## Approach

1. Raw data exploration. Build dictionary from corpus. 
2. Text data pre-processing. Split data into training, validation and test set.     
3. Build a LSTM model that output sentiment data from pre-processed data.
4. Tweak hyper-parameters. Train the model with different graph and training configuration.  
5. visualize the training process and result in Jupyter Notebook and TensorBoard.
4. Create Flask Rest API to perform data pre-processing and feed data into classification model.


