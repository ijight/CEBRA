import os
import numpy as np
import torch
from cebra import CEBRA
from cebra.data.datasets import TensorDataset, DatasetCollection

# Define the path to the data cache folder where the saved files are located
data_cache_folder = "cebra/data-cache/"
holdout_session_ids = ["710504563", "623339221", "589441079", "603763073", "676503588", "652092676", 
                       "649409874", "671164733", "623347352", "649401936", "555042467", "646016204", 
                       "595273803", "539487468", "637669284", "539497234", "652737678", "654532828", 
                       "669233895", "560926639", "547388708", "595806300", "689388034", "649938038", 
                       "645689073", "510514474", "505695962", "512326618", "562122508", "653122667"]

# Paths for saving and loading preprocessed data
processed_data_path = "processed_data.pt"
processed_labels_path = "processed_labels.pt"
processed_names_path = "processed_names.pt"
sorted_session_order_path = "sorted_session_order.txt"  # To save the sorted order of session names

# Check if preprocessed data exists
if os.path.exists(processed_data_path) and os.path.exists(processed_labels_path):
    # Load the preprocessed data to avoid reprocessing
    print("Loading preprocessed data...")
    datas = torch.load(processed_data_path)
    labels = torch.load(processed_labels_path)
    names = torch.load(processed_names_path)
else:
    # Prepare a list for storing datasets for each session
    datas = []
    labels = []
    names = []

    # Get the sorted list of files in the data-cache folder
    file_list = sorted([f for f in os.listdir(data_cache_folder) if f.endswith('.npz')])

    # Iterate over the sorted list of .npz files
    for file_name in file_list:
        print(f"Loading file: {file_name}")
        session_id = file_name.split('.')[0]

        # Load the data from the .npz file
        file_path = os.path.join(data_cache_folder, file_name)
        data = np.load(file_path)

        # Extract the saved variables from the .npz file
        neural_data = data['neural_data']  # shape: (n_neurons, total_timepoints)
        orientations = data['orientations']  # (n_trials,)
        temporal_frequencies = data['temporal_frequencies']  # (n_trials,)
        experiment_id = data['experiment_id']  # Experiment ID
        mask = data["blank_mask"].astype(bool)  # Mask of blank trials
        starts = data['starts']  # Start timepoints for each trial
        ends = data['ends']  # End timepoints for each trial

        trial_neural_data = []
        trial_orientations = []
        trial_temporal_frequencies = []

        # Iterate through each trial
        for i, (start, end) in enumerate(zip(starts, ends)):
            if mask[i]:  # Skip blank trials
                continue

            start, end = int(start), int(end)  # Get the start and end timepoints for each trial
            trial_data = neural_data[:, start:end]  # Extract the neural data for the trial

            # Append the trial data, orientations, and frequencies to the lists
            trial_neural_data.append(torch.FloatTensor(trial_data.T))  # Transpose to (timepoints, neurons)
            trial_orientations.append(torch.FloatTensor([orientations[i]] * (end - start)))  # Repeat orientation
            trial_temporal_frequencies.append(torch.FloatTensor([temporal_frequencies[i]] * (end - start)))  # Repeat frequency

        # Concatenate trials along the time axis
        neural_data_tensor = torch.cat(trial_neural_data, dim=0)  # Concatenate neural data
        orientations_behavior = torch.cat(trial_orientations, dim=0).unsqueeze(1)  # Concatenate and add dimension
        temporal_frequencies_behavior = torch.cat(trial_temporal_frequencies, dim=0).unsqueeze(1)  # Concatenate and add dimension

        # Combine behavior data (orientations and temporal frequencies)
        combined_behavior = torch.cat((orientations_behavior, temporal_frequencies_behavior), dim=1)

        print(neural_data_tensor.shape)
        print(combined_behavior.shape)
        assert neural_data_tensor.shape[0] == combined_behavior.shape[0], "Mismatch in neural data and behavior length"

        names.append(experiment_id)
        datas.append(neural_data_tensor)
        labels.append(combined_behavior)
        print(f"Skipping holdout session: {session_id}")

    # Save the processed data for future runs
    torch.save(datas, processed_data_path)
    torch.save(labels, processed_labels_path)
    torch.save(names, processed_names_path)
    print(f"Processed data saved to {processed_data_path} and {processed_labels_path}.")

# Initialize and train the CEBRA model
multi_cebra_model = CEBRA(
    model_architecture='offset10-model',
    batch_size=64,
    learning_rate=3e-4,
    temperature=1,
    output_dimension=3,
    max_iterations=100,
    distance='cosine',
    conditional='time_delta',
    device='cuda_if_available',
    verbose=True,
    time_offsets=10
)

# Save a sorted order of session names for future use
if not os.path.exists(sorted_session_order_path):
    with open(sorted_session_order_path, "w") as f:
        for session_id in names:
            f.write(str(session_id) + "\n")

print(len(datas), len(labels))
multi_cebra_model.fit(datas)

# Save the trained model
multi_cebra_model.save("multi_session_model_unsupervised.pt")
print("Model saved as 'multi_session_model.pt'")
