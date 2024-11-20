import os
import numpy as np
import torch
from cebra import CEBRA, L1LinearRegressor
from sklearn.metrics import mean_squared_error

holdout_session_ids = ["710504563", "623339221", "589441079", "603763073", "676503588", "652092676", 
                       "649409874", "671164733", "623347352", "649401936", "555042467", "646016204", 
                       "595273803", "539487468", "637669284", "539497234", "652737678", "654532828", 
                       "669233895", "560926639", "547388708", "595806300", "689388034", "649938038", 
                       "645689073", "510514474", "505695962", "512326618", "562122508", "653122667"]

# Paths for saving and loading preprocessed data
MODEL = "temporal_frequencies"
model_path = f"multi_session_model_{MODEL}.pt"
sorted_session_order_path = f"sorted_session_order_{MODEL}.txt"

processed_data_path = f"processed_data_{MODEL}.pt"
processed_labels_path = f"processed_labels_{MODEL}.pt"
processed_names_path = f"processed_names_{MODEL}.pt"

test_data_path = f"test_data_{MODEL}.pt"
test_labels_path = f"test_labels_{MODEL}.pt"
test_names_path = f"test_names_{MODEL}.pt"

# Load sorted session order
with open(sorted_session_order_path, "r") as f:
    sorted_session_order = [line.strip() for line in f.readlines()]

if os.path.exists(processed_data_path) and os.path.exists(processed_labels_path):
    # Load the preprocessed data to avoid reprocessing
    print("Loading preprocessed data...")
    datas = torch.load(processed_data_path)  # List of train data from each session
    labels = torch.load(processed_labels_path)  # List of train labels from each session
    names = torch.load(processed_names_path)  # List of session names from train

    test_datas = torch.load(test_data_path)  # List of test data from each session
    test_labels = torch.load(test_labels_path)  # List of test labels from each session
    test_names = torch.load(test_names_path)  # List of session names from test

name_to_index = {str(name): sorted_session_order.index(str(name)) for name in names}

# Get the embeddings for training data
train_embeddings_list = []
train_labels_list = []

loaded_model = CEBRA.load(model_path)

for session_data, session_label, session_id in zip(datas, labels, names):
    if session_id in holdout_session_ids:  # Don't train decoder on holdout sessions
        continue

    session_embedding = loaded_model.transform(session_data, name_to_index[str(session_id)])  # Transform each session
    session_embedding_tensor = torch.tensor(session_embedding)

    train_embeddings_list.append(session_embedding_tensor)
    train_labels_list.append(session_label)

# Concatenate lists into tensors
train_embeddings = torch.cat(train_embeddings_list, dim=0)
train_labels = torch.cat(train_labels_list, dim=0)

# Reshape train_labels to 1D array
train_labels = train_labels.view(-1)

# Initialize and train the L1LinearRegressor
regressor = L1LinearRegressor(alpha=0.1)
print("Training L1LinearRegressor...")
regressor.fit(train_embeddings, train_labels)

# Extract embeddings for the holdout data
holdout_embeddings_list = []
holdout_labels_list = []

# Transform the holdout data using the trained model
for holdout_data, holdout_label, session_id in zip(test_datas, test_labels, test_names):
    holdout_embedding = loaded_model.transform(holdout_data, name_to_index[str(session_id)])  # Transform each holdout session
    holdout_embeddings_list.append(torch.tensor(holdout_embedding))
    holdout_labels_list.append(holdout_label)

# Concatenate holdout data
holdout_embeddings = torch.cat(holdout_embeddings_list, dim=0)
holdout_labels = torch.cat(holdout_labels_list, dim=0)

# Reshape holdout_labels to 1D array
holdout_labels = holdout_labels.view(-1)

# Inference using the L1LinearRegressor
predictions = regressor.predict(holdout_embeddings)

# Calculate mean squared error for the holdout set
mse = mean_squared_error(holdout_labels.cpu().numpy(), predictions.cpu().numpy())
print(f"Holdout L1LinearRegressor mean squared error: {mse:.4f}")

# Optionally, calculate the MSE for each session separately
mse_list = []
for i in range(len(holdout_labels_list)):
    mse = mean_squared_error(holdout_labels_list[i].cpu().numpy(), predictions[i].cpu().numpy())
    mse_list.append(mse)
    print(f"MSE for session {test_names[i]}: {mse:.4f}")

# Print the mean MSE across all holdout sessions
print(f"Mean MSE: {np.mean(mse_list):.4f}")
