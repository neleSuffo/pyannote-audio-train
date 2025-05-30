import pandas as pd
import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import parselmouth # For Praat's pitch extraction
from sklearn.impute import SimpleImputer # For handling NaNs
import argparse
import joblib

# --- Configuration ---
AUDIO_DIR = "/home/nele_pauline_suffo/ProcessedData/childlens_audio"
# --- Training Mode Configuration ---
TRAIN_RTTM_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/complete.rttm"
MODEL_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/speech_classifier_pipeline.pkl'
IMPUTER_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/speech_feature_imputer.pkl'

# --- Paths for saving internal train/test splits from training mode ---
TRAIN_SPLIT_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/train_split_internal.rttm'
TEST_SPLIT_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/test_split_internal_for_eval.rttm'

# --- Evaluate Mode Configuration ---
EVAL_RTTM_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/test.rttm" # RTTM to evaluate model on
OUTPUT_CSV_PATH = "/home/nele_pauline_suffo/projects/pyannote-audio-train/final_classifications.csv"

RTTM_COLUMNS = ['type', 'file_id', 'channel', 'start_time', 'duration', 
                'NA1', 'NA2', 'diarization_label', 'NA3', 'NA4']
EXPECTED_LABELS = ["OHS", "CDS"] # Labels to be learned by the classifier

# --- 1. Parse RTTM ---
def parse_rttm(rttm_content):
    lines = rttm_content.strip().split('\n')
    data = []
    for line in lines:
        parts = line.split()
        if parts[0] == "SPEAKER" and len(parts) >= 8:
            try:
                start = float(parts[3])
                duration = float(parts[4])
                label = parts[7]
                if label in EXPECTED_LABELS: 
                    data.append([parts[0], parts[1], int(parts[2]), start, duration, 
                                 parts[5], parts[6], label, parts[8] if len(parts) > 8 else None, parts[9] if len(parts) > 9 else None])
            except ValueError as e:
                print(f"Skipping line due to parsing error (start/duration/channel): {line} - {e}")
            except IndexError as e:
                print(f"Skipping line due to missing parts: {line} - {e}")
    df = pd.DataFrame(data, columns=RTTM_COLUMNS)
    return df

# --- 2. Audio Loading and Slicing ---
def load_audio_segment(audio_path, start_time, duration):
    try:
        y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration)
        if len(y) == 0:
            print(f"Warning: Loaded empty audio segment from {audio_path} at {start_time} for {duration}s.")
            return None, None
        return y, sr
    except Exception as e:
        print(f"Error loading audio segment {audio_path}: {e}")
        return None, None

# --- 3. Feature Extraction ---
def extract_prosodic_features(y, sr):
    if y is None or len(y) == 0 or sr is None:
        num_mfcc = 13
        # Return NaNs for all expected features if audio is invalid
        return [np.nan] * (4 + 2 + 2 * num_mfcc) 
    
    features = []
    
    # Pitch features
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        pitch = sound.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames (zeros)
        if len(pitch_values) > 0:
            features.extend([np.mean(pitch_values), np.std(pitch_values), np.min(pitch_values), np.max(pitch_values)])
        else:
            features.extend([np.nan, np.nan, np.nan, np.nan]) # All NaNs if no voiced pitch found
    except Exception as e:
        print(f"Parselmouth pitch extraction error: {e}")
        features.extend([np.nan, np.nan, np.nan, np.nan])
        
    # RMS energy features
    rms = librosa.feature.rms(y=y)[0]
    if len(rms) > 0:
        features.extend([np.mean(rms), np.std(rms)])
    else:
        features.extend([np.nan, np.nan])
        
    # MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if mfccs.size > 0 and mfccs.shape[1] > 0: # Check if MFCCs were actually computed
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
    else:
        features.extend([np.nan] * 13 * 2) # 13 mean MFCCs + 13 std MFCCs
        
    return features

# --- 4. Data Preparation for Classifier Training ---
def prepare_classifier_data(rttm_df, audio_base_path):
    """
    Prepares feature vectors (X) and true labels (y) for training 
    the speech classifier (KCHI, OHS, CDS).
    """
    feature_vectors = []
    true_labels = []
    processed_segments_info = [] 

    print("\nPreparing data for speech classifier training...")
    for index, row in rttm_df.iterrows():
        if row['diarization_label'] in EXPECTED_LABELS:
            # Ensure file_id does not already contain .wav
            file_name = row['file_id']
            if not file_name.lower().endswith('.wav'):
                file_name += ".wav"
            segment_audio_path = os.path.join(audio_base_path, file_name)
            
            if not os.path.exists(segment_audio_path):
                print(f"Audio file not found: {segment_audio_path}. Skipping segment for training data.")
                continue

            y, sr = load_audio_segment(segment_audio_path, row['start_time'], row['duration'])
            if y is None:
                print(f"Could not load audio for segment: {row['file_id']} {row['start_time']}. Skipping.")
                continue

            features = extract_prosodic_features(y, sr)
            
            if features is not None: # extract_prosodic_features always returns a list
                feature_vectors.append(features)
                true_labels.append(row['diarization_label'])
                processed_segments_info.append(row.to_dict())
            else: # Should not happen if extract_prosodic_features is robust
                print(f"Feature extraction failed for segment {row['file_id']} {row['start_time']}. Skipping.")

    if not feature_vectors:
         print("No valid feature vectors collected for training. Aborting.")
         return None, None, None
         
    return np.array(feature_vectors), np.array(true_labels), processed_segments_info

# --- 5. Training the Speech Classifier ---
def train_speech_classifier(X, y, processed_segments_info, train_rttm_save_path, test_rttm_save_path):
    """
    Trains a classifier.
    X: Feature vectors.
    y: True labels.
    processed_segments_info: List of dicts containing original segment info for RTTM saving.
    train_rttm_save_path: Path to save the RTTM of the internal training split.
    test_rttm_save_path: Path to save the RTTM of the internal test split.
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("Cannot train classifier: No data provided.")
        return None, None
        
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)

    if X_imputed.shape[0] < 2 or len(np.unique(y)) < 2: # Need at least 2 samples and 2 classes
        print("Not enough samples or classes to train a meaningful speech classifier after imputation.")
        print(f"Samples: {X_imputed.shape[0]}, Unique classes: {np.unique(y)}")
        return None, None

    # Create an array of indices to split along with the data
    indices = np.arange(X_imputed.shape[0])
    
    try:
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
            X_imputed, y, indices, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train_test_split (likely due to insufficient samples for a class for stratification): {e}")
        print(f"Class distribution in y: {pd.Series(y).value_counts().to_dict()}")
        if len(np.unique(y)) == 1 and len(y) > 1:
             print("Only one class present. Training on all data, no test split for evaluation here.")
             X_train, X_test, y_train, y_test = X_imputed, np.array([]), y, np.array([])
             indices_train, indices_test = indices, np.array([]) # All data for train, none for test
        elif len(y) > 1 :
            print("Attempting train_test_split without stratification.")
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
                X_imputed, y, indices, test_size=0.25, random_state=42
            )
        else:
            print("Not enough data to perform train/test split.")
            return None, None
    
    # Save the RTTM files for the splits
    if processed_segments_info:
        train_segments_to_save = [processed_segments_info[i] for i in indices_train]
        test_segments_to_save = [processed_segments_info[i] for i in indices_test]
        
        if train_segments_to_save:
            write_rttm_from_segments(train_segments_to_save, train_rttm_save_path)
            print(f"Internal training split RTTM saved to: {train_rttm_save_path}")
        if test_segments_to_save:
            write_rttm_from_segments(test_segments_to_save, test_rttm_save_path)
            print(f"Internal test split RTTM saved to: {test_rttm_save_path}")

    print(f"Training speech classifier with {len(X_train)} samples, testing with {len(X_test)} samples.")
    print(f"Training labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
    if len(y_test) > 0:
        print(f"Test labels distribution: {pd.Series(y_test).value_counts().to_dict()}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced')) # class_weight='balanced' can help with imbalanced datasets
    ])

    pipeline.fit(X_train, y_train)

    if len(X_test) > 0 and len(y_test) > 0:
        print("\nSpeech Classifier Performance on Test Set:")
        y_pred_test = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred_test, zero_division=0, labels=EXPECTED_LABELS))
    else:
        print("\nNo test data to evaluate performance.")
    
    return pipeline, imputer

# --- 6. Apply Full Pipeline (for apply mode) ---
def apply_full_pipeline(rttm_df, audio_base_path, trained_speech_model, feature_imputer):
    final_classifications = []
    if trained_speech_model is None or feature_imputer is None:
        print("Trained model or imputer not available. Cannot perform full classification.")
        for index, row in rttm_df.iterrows():
            final_classifications.append({
                **row.to_dict(),
                'predicted_speech_class': 'MODEL_MISSING'
            })
        return pd.DataFrame(final_classifications)

    print(f"\nApplying model to {len(rttm_df)} segments...")
    for index, row in rttm_df.iterrows():
        file_name = row['file_id']
        if not file_name.lower().endswith('.wav'):
            file_name += ".wav"
        segment_audio_path = os.path.join(audio_base_path, file_name)
        
        classification_result = "ERROR_UNKNOWN" # Default status

        if not os.path.exists(segment_audio_path):
            classification_result = "ERROR_NO_AUDIO"
            print(f"Audio file not found: {segment_audio_path}. Skipping segment.")
        else:
            y, sr = load_audio_segment(segment_audio_path, row['start_time'], row['duration'])
            if y is None:
                classification_result = "ERROR_LOAD_AUDIO"
            else:
                features = extract_prosodic_features(y, sr)
                features_reshaped = np.array(features).reshape(1, -1)
                
                try:
                    features_imputed = feature_imputer.transform(features_reshaped)
                    if np.isnan(features_imputed).any(): # Check for NaNs after imputation
                        print(f"Warning: NaNs found in features after imputation for {row['file_id']} at {row['start_time']}. This might indicate issues with the segment or imputer training.")
                        classification_result = "ERROR_NAN_FEATURES_POST_IMPUTE"
                    else:
                        classification_result = trained_speech_model.predict(features_imputed)[0]
                except Exception as e:
                    print(f"Error during feature transformation or prediction for {row['file_id']} at {row['start_time']}: {e}")
                    classification_result = "ERROR_PREDICTION"
        
        final_classifications.append({
            'file_id': row['file_id'],
            'start_time': row['start_time'],
            'duration': row['duration'],
            'original_diarization_label': row.get('diarization_label', 'NA'), 
            'predicted_speech_class': classification_result
        })
    return pd.DataFrame(final_classifications)

def read_rttm_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: RTTM file not found at {file_path}")
        return None

def write_rttm_from_segments(segments_list, file_path):
    """
    Writes a list of segment dictionaries to an RTTM file.
    Each segment dictionary should conform to the RTTM_COLUMNS structure.
    """
    try:
        with open(file_path, 'w') as f:
            for seg in segments_list:
                # Ensure all RTTM fields are present, providing defaults for NA fields if missing
                line_parts = [
                    seg.get('type', 'SPEAKER'),
                    seg.get('file_id', 'UnknownFileID'),
                    str(seg.get('channel', 1)),
                    f"{seg.get('start_time', 0.0):.3f}",
                    f"{seg.get('duration', 0.0):.3f}",
                    seg.get('NA1', '<NA>'),
                    seg.get('NA2', '<NA>'),
                    seg.get('diarization_label', 'UnknownLabel'),
                    seg.get('NA3', '<NA>'),
                    seg.get('NA4', '<NA>') # Ensure 10 fields
                ]
                f.write(" ".join(line_parts) + "\n")
    except Exception as e:
        print(f"Error writing RTTM file {file_path}: {e}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, evaluate, or apply a speech classifier.")
    parser.add_argument("--mode", choices=['train', 'evaluate', 'apply'], required=True, help="Mode of operation: 'train', 'evaluate', or 'apply'.")
    # Arguments for 'apply' mode
    parser.add_argument("--input_rttm_apply", type=str, help="Path to the RTTM file for 'apply' mode.")
    parser.add_argument("--output_csv_apply", type=str, help="Path to save the CSV output for 'apply' mode (default: applied_classifications.csv).")
    args = parser.parse_args()

    if args.mode == 'train':
        print("--- Running in TRAIN mode ---")
        rttm_content = None
        if os.path.exists(TRAIN_RTTM_FILE_PATH):
            print(f"Loading RTTM from: {TRAIN_RTTM_FILE_PATH}")
            rttm_content = read_rttm_file(TRAIN_RTTM_FILE_PATH)
        else:
            print(f"Error: Training RTTM file not found at {TRAIN_RTTM_FILE_PATH}")

        if rttm_content is None:
            print("No RTTM data available for training. Exiting.")
            exit()

        rttm_df = parse_rttm(rttm_content)
        if rttm_df.empty:
            print("Parsed RTTM is empty for training. No data to process. Check RTTM content and EXPECTED_LABELS.")
            exit()
        
        print(f"\nParsed RTTM data for training ({len(rttm_df)} segments with expected labels)")

        X_features, y_labels, processed_segments = prepare_classifier_data(rttm_df, AUDIO_DIR)

        trained_model = None
        trained_imputer = None 

        if X_features is not None and y_labels is not None and len(X_features) > 0 and processed_segments is not None:
            print(f"\n--- Training Speech Classifier ({', '.join(EXPECTED_LABELS)}) ---")
            trained_model, trained_imputer = train_speech_classifier(
                X_features, y_labels, processed_segments, TRAIN_SPLIT_SAVE_PATH, TEST_SPLIT_SAVE_PATH
            )
            if trained_model and trained_imputer:
                print("\nClassifier training complete.")
                joblib.dump(trained_model, MODEL_SAVE_PATH)
                joblib.dump(trained_imputer, IMPUTER_SAVE_PATH)
                print(f"Trained model saved to {MODEL_SAVE_PATH}")
                print(f"Trained imputer saved to {IMPUTER_SAVE_PATH}")
            else:
                print("\nClassifier training failed or was skipped.")
        else:
            print("\nSkipping classifier training due to lack of data or errors during preparation.")

    elif args.mode == 'evaluate':
        print("--- Running in EVALUATE mode ---")
        rttm_content_eval = None
        if os.path.exists(EVAL_RTTM_FILE_PATH):
            print(f"Loading RTTM for evaluation from: {EVAL_RTTM_FILE_PATH}")
            rttm_content_eval = read_rttm_file(EVAL_RTTM_FILE_PATH)
        else:
            print(f"Error: RTTM file for evaluation not found at {EVAL_RTTM_FILE_PATH}")
        
        if rttm_content_eval is None:
            print("No RTTM data available for evaluation. Exiting.")
            exit()

        rttm_df_eval = parse_rttm(rttm_content_eval, labels_to_consider=EXPECTED_LABELS)
        if rttm_df_eval.empty:
            print("Parsed RTTM for evaluation is empty. No data to process. Check RTTM content and EXPECTED_LABELS.")
            exit()
        
        print(f"\nParsed RTTM data for evaluation ({len(rttm_df_eval)} segments with expected labels)")
        
        # Load trained model and imputer
        loaded_model = None
        loaded_imputer = None
        if os.path.exists(MODEL_SAVE_PATH):
            loaded_model = joblib.load(MODEL_SAVE_PATH)
            print(f"Loaded trained model from {MODEL_SAVE_PATH}")
        else:
            print(f"Error: Trained model file not found at {MODEL_SAVE_PATH}")
        
        if os.path.exists(IMPUTER_SAVE_PATH):
            loaded_imputer = joblib.load(IMPUTER_SAVE_PATH)
            print(f"Loaded feature imputer from {IMPUTER_SAVE_PATH}")
        else:
            print(f"Error: Feature imputer file not found at {IMPUTER_SAVE_PATH}")

        if loaded_model is None or loaded_imputer is None:
            print("Cannot proceed with evaluation due to missing model or imputer. Exiting.")
            exit()

        eval_df = apply_full_pipeline(rttm_df_eval, AUDIO_DIR, loaded_model, loaded_imputer)
        
        if not eval_df.empty:
            # Ensure 'original_diarization_label' and 'predicted_speech_class' columns exist
            if 'original_diarization_label' in eval_df.columns and 'predicted_speech_class' in eval_df.columns:
                y_true = eval_df['original_diarization_label']
                y_pred = eval_df['predicted_speech_class']
                
                print("\n--- Evaluation Metrics ---")
                valid_indices = y_true.isin(EXPECTED_LABELS)
                y_true_filtered = y_true[valid_indices]
                y_pred_filtered = y_pred[valid_indices]

                if not y_true_filtered.empty:
                    print(classification_report(y_true_filtered, y_pred_filtered, labels=EXPECTED_LABELS, zero_division=0))
                else:
                    print("No valid true labels found matching EXPECTED_LABELS for metric calculation.")

            else:
                print("Warning: 'original_diarization_label' or 'predicted_speech_class' column missing in the output DataFrame. Cannot calculate metrics.")

            try:
                eval_df.to_csv(OUTPUT_CSV_PATH, index=False)
                print(f"\nEvaluation results saved to {OUTPUT_CSV_PATH}")
            except Exception as e:
                print(f"Error saving evaluation results to CSV: {e}")
        else:
            print("Evaluation resulted in an empty DataFrame. No results to save or analyze.")

    else:
        print(f"Unknown mode: {args.mode}. Choose 'train' or 'evaluate'.")