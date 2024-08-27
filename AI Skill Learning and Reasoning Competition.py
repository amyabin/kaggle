import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the data
def load_json(filename):
    with open(filename) as f:
        return json.load(f)

# Load datasets (assumes data is available in the competition environment)
training_challenges = load_json('/kaggle/input/unzipdata/arc-agi_training_challenges.json')
evaluation_challenges = load_json('/kaggle/input/unzipdata/arc-agi_evaluation_challenges.json')
test_challenges = load_json('/kaggle/input/unzipdata/arc-agi_test_challenges.json')

def preprocess_grid(grid, max_length=None):
    if isinstance(grid, int):
        return [grid] if max_length is None else [grid] + [0] * (max_length - 1)
    
    if not isinstance(grid, list):
        raise TypeError(f"Expected list or int, got {type(grid)}")
    
    flat_list = []
    for sublist in grid:
        if isinstance(sublist, list):
            flat_list.extend(sublist)
        else:
            if isinstance(sublist, int):
                flat_list.append(sublist)
            else:
                raise TypeError(f"Expected list or int, got {type(sublist)} in grid")
    
    if max_length is not None:
        return flat_list[:max_length] + [0] * max(0, max_length - len(flat_list))
    
    return flat_list

def preprocess_data(challenges):
    X = []
    y = []

    max_input_length = 0
    max_output_length = 0

    if isinstance(challenges, dict):
        for key, task in challenges.items():
            try:
                if 'train' in task and isinstance(task['train'], list):
                    for train_pair in task['train']:
                        if 'input' in train_pair and 'output' in train_pair:
                            input_grid = train_pair['input']
                            output_grid = train_pair['output']
                            X.append(preprocess_grid(input_grid))
                            y.append(preprocess_grid(output_grid))
                            max_input_length = max(max_input_length, len(preprocess_grid(input_grid)))
                            max_output_length = max(max_output_length, len(preprocess_grid(output_grid)))
                else:
                    print(f"Skipping task due to unexpected 'train' format: {task}")
            except Exception as e:
                print(f"Error processing task: {task}, Error: {e}")

    X_array = np.array([preprocess_grid(grid, max_input_length) for grid in X])
    y_array = np.array([preprocess_grid(grid, max_output_length) for grid in y])

    return X_array, y_array, max_input_length, max_output_length

# Preprocess the data
X_train, y_train, max_input_length, max_output_length = preprocess_data(training_challenges)
X_eval, y_eval, _, _ = preprocess_data(evaluation_challenges)

# Train the model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000))
])
pipeline.fit(X_train, y_train)

# Prepare test data
def prepare_test_data(test_challenges, max_length):
    X_test = []
    if isinstance(test_challenges, dict):
        for key, task in test_challenges.items():
            try:
                if 'test' in task and isinstance(task['test'], list):
                    for test_pair in task['test']:
                        if 'input' in test_pair:
                            X_test.append(preprocess_grid(test_pair['input'], max_length))
                else:
                    print(f"Skipping task due to unexpected 'test' format: {task}")
            except Exception as e:
                print(f"Error processing test task: {task}, Error: {e}")

    return np.array(X_test)

# Prepare predictions
def generate_predictions(test_challenges, pipeline, max_length):
    X_test = prepare_test_data(test_challenges, max_length)
    predictions = pipeline.predict(X_test)
    
    submission = {}
    idx = 0
    
    for key, task in test_challenges.items():
        num_outputs = len(task.get('test', []))
        task_predictions = []
        
        for _ in range(num_outputs):
            # Check the shape of predictions
            pred_shape = predictions[idx].shape
            num_elements = predictions[idx].size
            
            # Determine the shape based on the number of elements and test input size
            num_rows = len(test_challenges[key]['test'][0]['input'])
            num_cols = num_elements // num_rows
            
            # Print out for debugging
            print(f"Task: {key}")
            print(f"Predictions Shape: {pred_shape}")
            print(f"Number of Rows: {num_rows}, Number of Columns: {num_cols}")

            try:
                attempt_1 = predictions[idx].reshape((num_rows, num_cols)).tolist()
                attempt_2 = predictions[idx].reshape((num_rows, num_cols)).tolist()  # Adjust this if necessary
            except ValueError as e:
                print(f"Reshape error for task {key}: {e}")
                attempt_1 = [[0] * num_cols] * num_rows
                attempt_2 = [[0] * num_cols] * num_rows

            task_predictions.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })
            idx += 1
        
        submission[key] = task_predictions

    return submission

# Generate and save predictions
submission = generate_predictions(test_challenges, pipeline, max_input_length)

with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f, indent=4)

print("Submission file created.")
