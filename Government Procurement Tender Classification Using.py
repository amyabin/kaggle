import pandas as pd

# Load the dataset
data = pd.read_csv('/kaggle/input/government-procurement-dataset/government-procurement-via-gebiz.csv')
data.head()
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Example preprocessing: cleaning text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

data['cleaned_description'] = data['tender_description'].apply(clean_text)

def label_category(description):
    if 'construction' in description:
        return 'Construction'
    elif 'medical' in description:
        return 'Medical'
    else:
        return 'Other'

data['Category'] = data['cleaned_description'].apply(label_category)

X = data['cleaned_description']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

model = RandomForestClassifier()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

print(classification_report(y_test, y_pred))


# Predict categories for the entire dataset
data_tfidf = vectorizer.transform(data['cleaned_description'])
data['Predicted_Category'] = model.predict(data_tfidf)

# Display the updated dataset
data.head()

# Filter the data to show only records with Predicted_Category other than 'Other'
non_other_data = data[data['Predicted_Category'] != 'Other']

# Display the filtered dataset
non_other_data.head()

# Write the filtered data to a new CSV file
non_other_data.to_csv('/kaggle/working/Predicted_Category_government-procurement-via-gebiz.csv', index=False)

