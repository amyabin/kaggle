import pandas as pd
import plotly.express as px

# Load the submission data
submission_df = pd.read_csv('submission.csv')

# Example: Distribution of Survival Predictions
fig = px.histogram(submission_df, x='Survived', title='Distribution of Survival Predictions')
fig.show()

# Example: Merge with original test data for more context
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
merged_df = pd.merge(test_df, submission_df, on='PassengerId')

# Example: Interactive plot of Survival Prediction by Age
fig = px.scatter(merged_df, x='Age', y='Survived', color='Sex', title='Survival Prediction by Age and Gender')
fig.show()
