import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from cebra import CEBRA
from sklearn.neighbors import KNeighborsClassifier
import csv
from tqdm import tqdm

# Holdout session IDs
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

N = 60

# Paths for saving and loading preprocessed data
MODEL = "both"  # Change to "both" to handle the dual-label case
model_path = f"multi_session_model_{MODEL}.pt"
sorted_session_order_path = f"sorted_session_order_{MODEL}.txt"

processed_data_path = f"processed_data_{MODEL}.pt"
processed_labels_path = f"processed_labels_{MODEL}.pt"
processed_names_path = f"processed_names_{MODEL}.pt"

test_data_path = f"test_data_{MODEL}.pt"
test_labels_path = f"test_labels_{MODEL}.pt"
test_names_path = f"test_names_{MODEL}.pt"

# Load the model and data
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
train_labels1_windows = []
train_labels2_windows = []

# Split the labels into two sets and slice the data into windows
for session_data, session_label, session_id in zip(datas, labels, names):
    if session_id in holdout_session_ids:  # Don't train decoder on holdout sessions
        continue
    
    if session_data.shape[0] < N:
        print(f"Skipping session {session_id} due to insufficient data")
        continue

    print(f"Processing session {session_id}...")

    session_embedding = loaded_model.transform(session_data, name_to_index[str(session_id)])  # Transform each session
    session_embedding_tensor = torch.tensor(session_embedding)
    session_label_tensor = torch.tensor(session_label)

    # Slice embeddings and split labels into two tasks
    sliced_embeddings, sliced_labels = slice_into_windows(session_embedding_tensor, session_label_tensor, N)
    sliced_labels1 = sliced_labels[:, :, 0]  # First label
    sliced_labels2 = sliced_labels[:, :, 1]  # Second label

    train_embeddings_windows.extend(sliced_embeddings)
    train_labels1_windows.extend(sliced_labels1)
    train_labels2_windows.extend(sliced_labels2)

# Flatten the windows and convert to numpy arrays
train_embeddings_windows = [emb.numpy().reshape(-1) for emb in train_embeddings_windows]
train_labels1_windows = [label.numpy().reshape(-1) for label in train_labels1_windows]
train_labels2_windows = [label.numpy().reshape(-1) for label in train_labels2_windows]

print(f"Training data shape: {np.array(train_embeddings_windows).shape}")
print(f"Training label1 shape: {np.array(train_labels1_windows).shape}")
print(f"Training label2 shape: {np.array(train_labels2_windows).shape}")

# Initialize and train two KNN classifiers (one for each label)
knn1 = KNeighborsClassifier(n_neighbors=60, metric="cosine")
knn2 = KNeighborsClassifier(n_neighbors=60, metric="cosine")

print("Training KNN Decoder for Label 1...")
knn1.fit(train_embeddings_windows, train_labels1_windows)

print("Training KNN Decoder for Label 2...")
knn2.fit(train_embeddings_windows, train_labels2_windows)

import matplotlib.pyplot as plt

# Initialize lists to store neuron counts and accuracies
neuron_counts = []
mean_accuracies = []

# Modify the inference loop to also store neuron counts and accuracies
print("Started inference on holdout sessions...")
with open(f"knn_predictions_{MODEL}_1.csv", "w", newline='') as csvfile:
    fieldnames = ['Session', 'True_Label1', 'Predicted_Label1', 'True_Label2', 'Predicted_Label2', 'Accuracy_Label1', 'Accuracy_Label2', 'Neuron_Count']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Iterate over the holdout sessions with a progress bar
    for holdout_data, holdout_label, holdout_session_id in tqdm(zip(test_datas, test_labels, test_names),
                                                               desc="Processing holdout sessions", total=len(test_datas)):
        # Transform the holdout session data using the loaded model
        holdout_embedding = loaded_model.transform(holdout_data, name_to_index[str(holdout_session_id)])
        holdout_embedding_tensor = torch.tensor(holdout_embedding)
        holdout_label_tensor = torch.tensor(holdout_label)

        # Slice the holdout session into windows
        holdout_embeddings_windows, holdout_labels_windows = slice_into_windows(holdout_embedding_tensor, holdout_label_tensor, N)

        # Split the holdout labels
        holdout_labels1_windows = holdout_labels_windows[:, :, 0]  # First label
        holdout_labels2_windows = holdout_labels_windows[:, :, 1]  # Second label

        # Get neuron count from the shape of the embeddings (number of features per window)
        neuron_count = holdout_data.shape[-1]
        print(f"Neuron count: {neuron_count}")
        neuron_counts.append(neuron_count)

        # Initialize lists to store predictions and true labels
        all_predictions1 = []
        all_true_labels1 = []
        all_predictions2 = []
        all_true_labels2 = []

        # Iterate over windows
        for i in tqdm(range(holdout_embeddings_windows.size(0)), desc=f"Windows in session {holdout_session_id}"):
            # Flatten the embeddings and labels
            window_embedding = holdout_embeddings_windows[i].numpy().reshape(-1)
            window_true_label1 = holdout_labels1_windows[i].numpy().reshape(-1)
            window_true_label2 = holdout_labels2_windows[i].numpy().reshape(-1)

            # Predict both labels
            prediction1 = knn1.predict([window_embedding])[0]
            prediction2 = knn2.predict([window_embedding])[0]

            # Store predictions and true labels
            all_predictions1.extend(prediction1)
            all_true_labels1.extend(window_true_label1)
            all_predictions2.extend(prediction2)
            all_true_labels2.extend(window_true_label2)

        # Calculate accuracy for both labels
        accuracy_label1 = accuracy_score(all_true_labels1, all_predictions1)
        accuracy_label2 = accuracy_score(all_true_labels2, all_predictions2)

        # Store the mean accuracy of both labels
        mean_accuracy = (accuracy_label1 + accuracy_label2) / 2
        mean_accuracies.append(mean_accuracy)

        # Save results to CSV
        writer.writerow({
            'Session': holdout_session_id,
            'True_Label1': all_true_labels1,
            'Predicted_Label1': all_predictions1,
            'True_Label2': all_true_labels2,
            'Predicted_Label2': all_predictions2,
            'Accuracy_Label1': accuracy_label1,
            'Accuracy_Label2': accuracy_label2,
            'Neuron_Count': neuron_count
        })

        print(f"Session {holdout_session_id} - Accuracy Label 1: {accuracy_label1:.4} - Accuracy Label 2: {accuracy_label2:.4}")

# Plot neuron count against mean accuracy
plt.figure(figsize=(10, 6))

# Plot mean accuracies
plt.plot(neuron_counts, mean_accuracies, label='Mean Accuracy', marker='o', linestyle='-', color='purple')

# Add labels and title
plt.xlabel('Neuron Count (Features of Original Time Series)')
plt.ylabel('Mean Accuracy')
plt.title(f'Mean Accuracy vs Neuron Count for {MODEL} Model')
plt.legend()

# Show the plot
plt.savefig(f"mean_accuracy_vs_neuron_count_{MODEL}.png")

# Print all session average accuracies for final review
avg_mean_accuracy = np.mean(mean_accuracies)
print(f"Average mean accuracy across all sessions: {avg_mean_accuracy:.4}")

print("Inference, plotting, and saving complete.")