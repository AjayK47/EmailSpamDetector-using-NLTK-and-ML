# EmailSpamDetector-using-NLTK-and-ML

## Overview

This project implements a simple email spam detection system using two machine learning models: Multinomial Naive Bayes (NB) and Support Vector Machine (SVM). It utilizes natural language processing techniques to preprocess input emails, extracts features using TF-IDF vectorization, and applies the trained models to classify emails as spam or non-spam.

### Prerequisites
1. Python 3.10

2. NLTK library (Natural Language Toolkit)

3. Scikit-learn library (Machine Learning in Python)

4. Joblib

## Installation
1. Clone the repository:

   ```bash
   git clone https://github.com/AjayK47/EmailSpamDetector-using-NLTK-and-ML.git 
   ``` 
2. Clone the repository:

   ```bash
    pip install pandas nltk scikit-learn joblib 
    ```
## Model training 

#### If you just want to test or use the models skip this step and download models provided in Git repo and link.

The training process involves:

- Importing the necessary libraries.
- Loading the dataset (spam.csv).
- Data preprocessing, including lemmatization and stop words removal.
- Using the TF-IDF vectorizer to convert text data into numerical form.
- Splitting the dataset into training and testing sets.
- Training the Multinomial Naive Bayes and Support Vector Machine models.


The testing process evaluates the accuracy of the models on a test set and provides a confusion matrix. Two models are implemented:

- Save models for future use after training using joblib library
- 'spam_detect_model_NB.joblib' (Multinomial Naive Bayes model)
- 'spam_detect_model_svm.jobl' (Support Vector Machine model)
-  'tfidf_vectorizer.joblib' (TF-IDF vectorizer)

### Model usage :

- Load the Trained models , if you havent trained download models from repo and provided link 

- Import Necessary libraries mentioned Pre-Requsites

- Give your E-mail and pre-process data , Follow steps in test_classifier.

## Model Prediction Process 
#### How it actually works :

Once the EmailSpamDetector models are trained, you can use them to make predictions on new data. Here is the process for predicting whether a given text is spam or not:

- **Text Pre-processing:**
   - The input text undergoes pre-processing, which includes:
     - Converting the text to lowercase.
     - Removing non-alphabetic characters and special symbols.
     - Tokenizing the text into words.
     - Lemmatizing each word to its base form.
     - Removing stop words.

- **TF-IDF Vectorization:**
   - The pre-processed text is transformed using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. This converts the text into a numerical representation suitable for machine learning models.

- **Model Prediction:**
   - The transformed data is fed into the trained machine learning models:
     - **Multinomial Naive Bayes (NB) Model:** This model is well-suited for text classification tasks.
     - **Support Vector Machine (SVM) Model:** This model is effective in creating a hyperplane to separate classes.

- **Sentence-Level Predictions:**
   - Each sentence in the input text is independently classified as 'spam' or 'ham' using both models.

- **Combining Predictions:**
   - Predictions from both models are combined for each sentence. If either model classifies a sentence as 'spam', it is considered 'spam' for the final decision.

- **Threshold Classification:**
   - The final classification for the entire paragraph or email is determined based on a threshold. The default threshold is set to 30% of sentences classified as 'spam'. If the percentage exceeds this threshold, the overall classification is 'spam'; otherwise, it's 'ham'.

- This prediction process allows you to apply the trained models to new data and classify whether it contains spam or legitimate content.

- Feel free to customize the threshold or incorporate additional steps based on the requirements of your project.


### Acknowledgments

- This project utilizes the following libraries:
  -  NLTK (Natural Language Toolkit)
  -  Scikit-learn (Machine Learning in Python)

- This project utilizes this Dataset From Kaggle:

   - [Spam email detection](https://www.kaggle.com/datasets/yousefmohamed20/spam-email-detection)
