# Scream Detection Project

The "Scream Detection Project" uses MFCC-based audio features and a Random Forest Classifier to detect screams from the audio recordings.
 It classifies audio into :
   a) Ambient sounds
   b) Screams
   c) Normal conversations


# Model Architecture and Approach

Used the Random Forest Classifier, a powerful ensemble learning method based on decision trees.
we choose Random Forest cause it is Simple, fast to train and predict and accurate for small to mid-sized datasets.
Input Features are the MFCC (Mel Frequency Cepstral Coefficients) extracted from .wav files
n_estimators=100: Uses 100 decision trees to make predictions, improving stability and accuracy.
random_state=42: Ensures reproducibility of results.
Output Classes: 0 - Ambient | 1 - Scream | 2 - Conversation


# Data Preprocessing

This step involves creating a suitable dataset or in other words, converting raw audio files into structured numerical data (MFCCs) suitable for training a machine learning model.
Used three publicly available datasets representing different sound classes taken from different platforms 
    *Ambient Sounds: From ESC-50 dataset
    *Screams: From NotScreaming collection
    *Conversations: From Mozilla Common Voice (English) corpus

For each audio file:Duration is trimmed to 5 seconds, Sample rate is set to 16,000 Hz (16kHz), Mono audio conversion, MFCCs: 13 Mel Frequency Cepstral Coefficients are extracted as to make it lightweight and efficient.

Each audio sample is assigned a label:0(Ambient), 1(Scream), 2(Conversation)

MFCC feature vectors and their corresponding labels are combined into a Pandas DataFrame. The full dataset is saved as a CSV file "combined_features.csv"

This dataset will be used for training and testing the machine learning model.
    

# Training & Testing Strategy

Dataset splitted into 80% for training and 20% for testing


# Evaluation Metrics

Accuracy:It is a basic evaluation metric that measures how often the model makes correct predictions (~87% on test set for our model)
Confusion Matrix: To visualize classification performance
Classification Report:It Includes precision, recall, F1-score for each class


# How to Reproduce

1)** Clone the Repo or download the code files
2)Ensure these libraries are installed on your machine with pip command :
   pip install numpy pandas matplotlib seaborn scikit-learn librosa sounddevice

3)Place your combined_features.csv in the correct path (or can use your own dataset)
4)Run model_train.py to train and save the model
5)Launch the Interface created by using Python_Tkinter to test live 
