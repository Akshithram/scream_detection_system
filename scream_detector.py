# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib 

# Loading the Preprocessed Dataset with the path
df = pd.read_csv(r'C:\Users\Akshithram\OneDrive\Desktop\scream_detector_project\scream_detector_project\data\combined_features.csv')

# Split features and labels
X = df.drop('label', axis=1)
# Class labels (0 - Ambient Sounds, 1 - Screams, 2 - Normal Conversations)
y = df['label'] 

# Train-Test Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Prediction & Evaluation 
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)


print(f"\n Accuracy: {acc * 100:.2f}%")
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ambient", "Scream", "Conversation"]))

# Visualizing the Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Ambient", "Scream", "Conversation"],
            yticklabels=["Ambient", "Scream", "Conversation"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Saving the Trained model
joblib.dump(clf, "scream_detector_model.pkl")
print("\n Model saved as scream_detector_model.pkl")
