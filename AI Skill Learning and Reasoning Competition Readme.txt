AI Skill Learning and Reasoning Competition Overview This project involves developing an AI system to efficiently learn new skills and solve open-ended problems using the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) benchmark. The goal is to improve AI's ability to generalize to new problems outside its training data, contributing towards advancements in Artificial General Intelligence (AGI).

Project Structure

Data Preparation prepare_test_data: This function prepares the test data for the AI model by processing the input grids into a suitable format.
preprocess_data: This function preprocesses training and evaluation data by converting grids into numpy arrays with appropriate padding and reshaping.

Model Pipeline pipeline: A machine learning model or pipeline that performs predictions on the prepared test data. This could be any scikit-learn compatible model or a custom pipeline.

Prediction Generation generate_predictions: This function generates predictions for the test challenges using the trained model pipeline. It handles reshaping and formatting the predictions into the required JSON structure for submission.

Submission Submission Format: The predictions are saved in a JSON file named submission.json. Each task in the evaluation set has exactly two predictions (attempt_1 and attempt_2) for each test input.

Prepare Data: Use prepare_test_data to convert test challenges into a format suitable for model prediction.

Preprocess Data: Preprocess training and evaluation data using preprocess_data.

Train Model: Train your model or pipeline with the preprocessed training data.

Generate Predictions: Use generate_predictions to make predictions on test challenges.

Create Submission: Save the predictions in submission.json in the required format.

Dependencies numpy scikit-learn (if using machine learning models) json Example Usage python Copy code

Example usage
test_challenges = load_test_challenges('test_challenges.json') pipeline = train_model(training_challenges) max_input_length = 21 submission = generate_predictions(test_challenges, pipeline, max_input_length)

with open('submission.json', 'w') as f: json.dump(submission, f, indent=4) Contributing Contributions are welcome! Please fork the repository and submit a pull request with your changes.

License This project is licensed under the MIT License - see the LICENSE file for details.

Contact For questions or further information, please contact Navalur Shoeb
