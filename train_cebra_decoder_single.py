import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from cebra import CEBRA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import tqdm

holdout_session_ids = ["710504563", "623339221", "589441079", "603763073", "676503588", "652092676", 
                       "649409874", "671164733", "623347352", "649401936", "555042467", "646016204", 
                       "595273803", "539487468", "637669284", "539497234", "652737678", "654532828", 
                       "669233895", "560926639", "547388708", "595806300", "689388034", "649938038", 
                       "645689073", "510514474", "505695962", "512326618", "562122508", "653122667"]


# Function to slice data into N-sized windows
def slice_into_windows(embeddings, labels, window_size):
    sliced_embeddings = []
    sliced_labels = []
    for i in range(0, embeddings.shape[0] - window_size + 1, window_size):
        sliced_embeddings.append(embeddings[i:i + window_size])
        sliced_labels.append(labels[i:i + window_size])
    return torch.stack(sliced_embeddings), torch.stack(sliced_labels)

# Set the window size
N = 60 # Example window size, adjust as needed

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

loaded_model = CEBRA.load(model_path)

# Load sorted session order
with open(sorted_session_order_path, "r") as f:
    sorted_session_order = [line.strip() for line in f.readlines()]

if os.path.exists(processed_data_path) and os.path.exists(processed_labels_path):
    # Load the preprocessed data
    print("Loading preprocessed data...")
    datas = torch.load(processed_data_path)  # List of train data from each session
    labels = torch.load(processed_labels_path)  # List of train labels from each session
    names = torch.load(processed_names_path)  # List of session names from train

    test_datas = torch.load(test_data_path)  # List of test data from each session
    test_labels = torch.load(test_labels_path)  # List of test labels from each session
    test_names = torch.load(test_names_path)  # List of session names from test

name_to_index = {str(name): sorted_session_order.index(str(name)) for name in names}

# Initialize storage for the windowed data
train_embeddings_windows = []
train_labels_windows = []

for session_data, session_label, session_id in zip(datas, labels, names):
    if session_id in holdout_session_ids:  # Don't train decoder on holdout sessions
        continue
    
    if session_data.shape[0] < N:
        print(f"Skipping session {session_id} due to insufficient data")
        continue

    print(f"Processing session {session_id}...")
    print(f"Session data shape: {session_data.shape}")
    print(f"Session label shape: {session_label.shape}")

    session_embedding = loaded_model.transform(session_data, name_to_index[str(session_id)])  # Transform each session
    session_embedding_tensor = torch.tensor(session_embedding)
    session_label_tensor = torch.tensor(session_label)

    print(f"Session embedding shape: {session_embedding_tensor.shape}")
    print(f"Session label shape: {session_label.shape}")

    # Slice into windows (N embeddings -> N labels)
    sliced_embeddings, sliced_labels = slice_into_windows(session_embedding_tensor, session_label_tensor, N)

    train_embeddings_windows.extend(sliced_embeddings)
    train_labels_windows.extend(sliced_labels)

print(train_embeddings_windows[0].shape)
print(train_labels_windows[0].shape)

print(len(train_embeddings_windows))
print(len(train_labels_windows))


train_embeddings_windows = [emb.numpy().reshape(-1) for emb in train_embeddings_windows]
train_labels_windows = [label.numpy().reshape(-1) for label in train_labels_windows]

print(f"Training data shape: {np.array(train_embeddings_windows).shape}")
print(f"Training label shape: {np.array(train_labels_windows).shape}")

# Initialize and train the KNNClassifier on the windowed data
knn = KNeighborsClassifier(n_neighbors=6, metric="cosine")
multi_output_knn = MultiOutputClassifier(knn, n_jobs=-1)



print("Training KNN Decoder on windowed data...")
multi_output_knn.fit(
    train_embeddings_windows,  # Training on windows of embeddings
    train_labels_windows       # Predicting windows of labels directly
)

from tqdm import tqdm

print("Started inference on holdout sessions...")
# Now handle the holdout sessions, also using windowed slicing
with open("knn_predictions.csv", "w", newline='') as csvfile:
    fieldnames = ['Session', 'True', 'Predictions', 'Accuracy']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over the holdout sessions with a progress bar
    for holdout_data, holdout_label, holdout_session_id in tqdm(zip(test_datas, test_labels, test_names),
                                                               desc="Processing holdout sessions", total=len(test_datas)):
        print(f"Processing holdout session {holdout_session_id}...")

        # Transform the holdout session data using the loaded model
        holdout_embedding = loaded_model.transform(holdout_data, name_to_index[str(holdout_session_id)])
        holdout_embedding_tensor = torch.tensor(holdout_embedding)
        holdout_label_tensor = torch.tensor(holdout_label)

        # Slice the holdout session into windows (N embeddings -> N labels)
        holdout_embeddings_windows, holdout_labels_windows = slice_into_windows(holdout_embedding_tensor, holdout_label_tensor, N)

        # Initialize lists to store all predictions and true labels for the session
        all_predictions = []
        all_true_labels = []

        # Iterate over windows with a progress bar
        for i in tqdm(range(holdout_embeddings_windows.size(0)), desc=f"Windows in session {holdout_session_id}"):
            # Flatten the embeddings and labels for this window
            window_embedding = holdout_embeddings_windows[i].numpy().reshape(-1)  # Flatten the window
            window_true_label = holdout_labels_windows[i].numpy().reshape(-1)  # Flatten the window

            # Predict the labels for the current window
            prediction = multi_output_knn.predict([window_embedding])  # Predict requires 2D input
            print(prediction.shape)

            # Store the predictions and true labels
            all_predictions.extend(prediction[0])  # Prediction shape (1, M), flatten it
            all_true_labels.extend(window_true_label)  # Already 1D

        # Calculate accuracy for the entire session
        session_accuracy = accuracy_score(all_true_labels, all_predictions)

        # Save results to CSV
        writer.writerow({
            'Session': holdout_session_id,
            'True': all_true_labels,
            'Predictions': all_predictions,
            'Accuracy': session_accuracy
        })

        print(f"Session {holdout_session_id} - Accuracy: {session_accuracy:.4f}")