import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the submission data
submission_df = pd.read_csv('submission.csv')

# Example: Distribution of Survival Predictions
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=submission_df)
plt.title('Distribution of Survival Predictions')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Example: Merge with original test data for more context
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
merged_df = pd.merge(test_df, submission_df, on='PassengerId')

# Example: Survival Prediction by Gender
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', hue='Sex', data=merged_df)
plt.title('Survival Prediction by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
