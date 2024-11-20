import os
import numpy as np
import torch
from cebra import CEBRA, KNNDecoder
from sklearn.metrics import accuracy_score  # To compute classification accuracy

# Step 1: Define paths and session IDs
data_cache_folder = "data-cache/"
holdout_session_ids = ["710504563", "623339221", "589441079", "603763073", "676503588", "652092676", 
                       "649409874", "671164733", "623347352", "649401936", "555042467", "646016204", 
                       "595273803", "539487468", "637669284", "539497234", "652737678", "654532828", 
                       "669233895", "560926639", "547388708", "595806300", "689388034", "649938038", 
                       "645689073", "510514474", "505695962", "512326618", "562122508", "653122667"]

# Step 2: Load and process the data session by session
for file_name in os.listdir(data_cache_folder):
    if file_name.endswith('.npz'):
        session_id = file_name.split('.')[0]
        file_path = os.path.join(data_cache_folder, file_name)
        data = np.load(file_path)

        # Extract variables from the .npz file
        neural_data = data['neural_data']  # shape: (n_neurons, total_timepoints)
        orientations = data['orientations']  # (n_trials,)
        temporal_frequencies = data['temporal_frequencies']  # (n_trials,)
        mask = data["blank_mask"].astype(bool)  # Mask of blank trials
        starts = data['starts']  # Start timepoints for each trial
        ends = data['ends']  # End timepoints for each trial

        trial_neural_data, trial_orientations, trial_temporal_frequencies = [], [], []

        # Step 3: Process the trials for each session
        for i, (start, end) in enumerate(zip(starts, ends)):
            if mask[i]:  # Skip blank trials
                continue

            start, end = int(start), int(end)
            trial_data = neural_data[:, start:end]
            trial_neural_data.append(torch.FloatTensor(trial_data.T))  # (timepoints, neurons)
            trial_orientations.append(torch.LongTensor([orientations[i]] * (end - start)))  # Use LongTensor for classification
            trial_temporal_frequencies.append(torch.LongTensor([temporal_frequencies[i]] * (end - start)))  # Use LongTensor

        if trial_neural_data:
            neural_data_tensor = torch.cat(trial_neural_data, dim=0)  # Concatenate trials
            orientations_tensor = torch.cat(trial_orientations, dim=0)
            temporal_frequencies_tensor = torch.cat(trial_temporal_frequencies, dim=0)

            # Step 4: Perform 80-20 train/test split
            split_index = int(neural_data_tensor.shape[0] * 0.8)
            train_data = neural_data_tensor[:split_index]
            test_data = neural_data_tensor[split_index:]

            train_orientations = orientations_tensor[:split_index]
            test_orientations = orientations_tensor[split_index:]
            train_temporal_frequencies = temporal_frequencies_tensor[:split_index]
            test_temporal_frequencies = temporal_frequencies_tensor[split_index:]

            # Step 5: Train a single CEBRA model for this session
            print(f"Training CEBRA model for session {session_id}")
            cebra_model = CEBRA(
                model_architecture='offset10-model',
                batch_size=64,
                learning_rate=3e-4,
                temperature=1,
                output_dimension=1,
                max_iterations=10000,
                distance='cosine',
                conditional='time_delta',
                device='cuda_if_available',
                verbose=True,
                time_offsets=10
            )

            # Fit the model on the training data
            cebra_model.fit([train_data], [train_orientations])  # We can choose orientations as primary behavior
            print(f"CEBRA model for session {session_id} trained.")

            # Step 6: Transform the training and test data to get embeddings
            train_embedding = cebra_model.transform(train_data, 0)  # Training embedding
            test_embedding = cebra_model.transform(test_data, 0)    # Test embedding

            # Step 7: Train the KNN decoder on the training embeddings and labels
            decoder_orientation = KNNDecoder(n_neighbors=36)  # Initialize KNN Decoder for orientations
            decoder_temporal_frequencies = KNNDecoder(n_neighbors=36)  # Initialize KNN Decoder for temporal frequencies

            decoder_orientation.fit(torch.FloatTensor(train_embedding), train_orientations)  # Fit the decoder on the orientations
            decoder_temporal_frequencies.fit(torch.FloatTensor(train_embedding), train_temporal_frequencies)  # Fit the decoder on the frequencies

            print(f"KNN decoders trained for session {session_id}.")

            # Step 8: Evaluate the decoder on the test embeddings (orientation and temporal frequencies separately)
            test_embedding_tensor = torch.FloatTensor(test_embedding)

            # Predict and score for orientations
            predicted_orientations = decoder_orientation.predict(test_embedding_tensor)
            orientation_accuracy = accuracy_score(test_orientations.cpu().numpy(), predicted_orientations.cpu().numpy())

            # Predict and score for temporal frequencies
            predicted_temporal_frequencies = decoder_temporal_frequencies.predict(test_embedding_tensor)
            temporal_frequency_accuracy = accuracy_score(test_temporal_frequencies.cpu().numpy(), predicted_temporal_frequencies.cpu().numpy())


            # Print results for each label type
            print(f"Orientation accuracy for session {session_id}: {orientation_accuracy:.4f}")
            print(f"Temporal frequency accuracy for session {session_id}: {temporal_frequency_accuracy:.4f}")

            # Optionally, print some of the predicted labels
            print(f"Predicted orientations for session {session_id}: {predicted_orientations[:10]}")
            print(f"Predicted temporal frequencies for session {session_id}: {predicted_temporal_frequencies[:10]}")
