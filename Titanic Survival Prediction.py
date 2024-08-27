import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

# Data Preprocessing
# Convert categorical variables to numeric
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Map 'Embarked' column to numeric values
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop columns that are not needed
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']  # Include 'PassengerId'
train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop, errors='ignore')  # Drop 'PassengerId' from test data if it exists

# Define features and target
X_train = train_df.drop(columns=['Survived'])
y_train = train_df['Survived']

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_train_split = imputer.fit_transform(X_train_split)
X_val_split = imputer.transform(X_val_split)

# Prepare test data for imputation and prediction
X_test = test_df.copy()  # Copy test_df to avoid altering the original data
X_test = imputer.transform(X_test)  # Impute missing values in test set

# Ensure columns in X_test match those in X_train
X_test = pd.DataFrame(X_test, columns=X_train.columns)  # Ensure columns are aligned

# Train a model
model = RandomForestClassifier()
model.fit(X_train_split, y_train_split)

# Make predictions on the validation set
y_pred = model.predict(X_val_split)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val_split, y_pred))
print("Classification Report:\n", classification_report(y_val_split, y_pred))

# Prepare test data for prediction
# Ensure the columns in X_test match those in X_train
# The columns are already aligned from previous steps

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Prepare submission file
# Re-load PassengerId to include in the submission
submission = pd.DataFrame({
    'PassengerId': pd.read_csv('/kaggle/input/titanic/test.csv')['PassengerId'],  # Include PassengerId for submission
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
