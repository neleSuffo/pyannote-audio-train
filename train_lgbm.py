import librosa
import os
import joblib
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import parselmouth
import soundfile as sf
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

# --- Configuration ---
AUDIO_DIR = "/home/nele_pauline_suffo/ProcessedData/childlens_audio"
# --- Training Mode Configuration ---
TRAIN_RTTM_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/complete.rttm"
MODEL_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/lgbm_classifier.pkl'
IMPUTER_SAVE_PATH = '/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/lgbm_imputer.pkl'

# --- Apply Mode Configuration ---
OUTPUT_CSV_PATH = "/home/nele_pauline_suffo/projects/pyannote-audio-train/final_classifications.csv"

RTTM_COLUMNS = ['type', 'file_id', 'channel', 'start_time', 'duration', 
                'NA1', 'NA2', 'diarization_label', 'NA3', 'NA4', "child_id"]
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
                                 parts[5], parts[6], label, parts[8] if len(parts) > 8 else None, parts[9] if len(parts) > 9 else None, parts[10] if len(parts) > 10 else None])
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
def prepare_lgbm_classifier_data(rttm_df, audio_base_path):
    """
    Prepares feature vectors (X) and true labels (y) for training 
    the classifier, using existing labels from the rttm_df.
    Assumes rttm_df is already filtered by parse_rttm to include only EXPECTED_LABELS.
    """
    feature_vectors = []
    true_labels = []
    group_ids = [] # To store group identifiers (child_id)

    print("\nPreparing data for LGBM classifier training...")
    for index, row in rttm_df.iterrows():
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
        
        current_label = row['diarization_label']

        # Only include segments with non-NaN features
        if not any(np.isnan(f) for f in features): # More robust NaN check for list of features
            feature_vectors.append(features)
            true_labels.append(current_label)
            group_ids.append(row['child_id']) # Add file_id as group identifier
        else:
            print(f"Segment {row['child_id']} at {row['start_time']}s has NaN features. Skipping.")

    if not feature_vectors:
        print("No valid feature vectors collected for LGBM training. Aborting.")
        return None, None, None # Adjusted return
    
    return np.array(feature_vectors), np.array(true_labels), np.array(group_ids) # Adjusted return

# --- 5. Training the Speech Classifier ---
def train_lgbm_classifier(X, y, groups):
    """
    Trains a LightGBM classifier with hyperparameter tuning,
    splitting data while keeping groups intact.
    X: Feature vectors.
    y: True labels
    groups: Group identifiers for each sample in X and y.
    """
    if X is None or y is None or len(X) == 0 or len(y) == 0:
        print("Cannot train classifier: No data provided.")
        return None, None

    # Impute NaNs
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X)

    if X_imputed.shape[0] < 2 or len(np.unique(y)) < 2:
        print("Not enough samples or classes to train a meaningful LGBM classifier after imputation.")
        return None, None

    # Split data using StratifiedGroupKFold
    if groups is None or len(groups) != X_imputed.shape[0]:
        print("Group information is missing or mismatched. Falling back to standard stratified split (groups not kept together).")
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.20, random_state=42, stratify=y
        )
    else:
        n_splits_for_group_split = 5  # For a test_size of 0.20 (1/5)
        sgkf = StratifiedGroupKFold(n_splits=n_splits_for_group_split, shuffle=True, random_state=42)
        
        try:
            # Get the first (and only needed) split
            train_idx, test_idx = next(sgkf.split(X_imputed, y, groups))
            
            X_train, X_test = X_imputed[train_idx], X_imputed[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            train_groups_set = set(groups[train_idx])
            test_groups_set = set(groups[test_idx])
            
            print(f"Data split using StratifiedGroupKFold: {len(X_train)} train samples, {len(X_test)} test samples.")
            print(f"Number of unique groups in train: {len(train_groups_set)}, in test: {len(test_groups_set)}")

            if not train_groups_set.isdisjoint(test_groups_set):
                print("Warning: Group overlap detected between train and test sets. This should not happen with StratifiedGroupKFold.")
            else:
                print("No group overlap between train and test sets, as expected.")

        except ValueError as e:
            print(f"Error during StratifiedGroupKFold split (e.g., a class is not present in enough groups for {n_splits_for_group_split} splits): {e}")
            print("Falling back to standard stratified split (groups not kept together).")
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=0.20, random_state=42, stratify=y
            )
    
    print(f"Training LGBM classifier with {len(X_train)} samples, testing with {len(X_test)} samples.")
    print(f"Training labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
    if len(y_test) > 0:
        print(f"Test labels distribution: {pd.Series(y_test).value_counts().to_dict()}")

    # Define pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1))
    ])

    # Define hyperparameter grid
    param_grid = {
        'classifier__learning_rate': [0.01, 0.05, 0.1],  # 3 values
        'classifier__num_leaves': [15, 31, 50],          # 3 values
        'classifier__max_depth': [3, 5, -1],             # 3 values
        'classifier__n_estimators': [50, 100, 200],      # 3 values
        'classifier__min_child_samples': [10, 20, 50],   # 3 values
        'classifier__subsample': [0.8, 1.0],             # 2 values
        'classifier__colsample_bytree': [0.8, 1.0],      # 2 values
        'classifier__scale_pos_weight': [1.0, 2.0]  # 2 values for class imbalance
    }

    # Total combinations: 3 × 3 × 3 × 3 × 3 × 2 × 2 × 2 = 486

    # Perform Grid Search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring='f1_macro',  # Optimize for balanced performance across classes
        cv=5,  # 5-fold cross-validation
        n_jobs=2,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # Print best parameters and score
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validation F1 Score:", grid_search.best_score_)

    # Evaluate on test set
    print("\nLGBM Classifier Performance on Test Set:")
    y_pred_test = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Feature importance
    best_model = grid_search.best_estimator_.named_steps['classifier']
    feature_names = ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
                     'rms_mean', 'rms_std'] + [f'mfcc_mean_{i}' for i in range(13)] + [f'mfcc_std_{i}' for i in range(13)]
    print("\nFeature Importance (LightGBM):")
    for name, importance in zip(feature_names, best_model.feature_importances_):
        print(f"{name}: {importance}")

    return grid_search.best_estimator_, imputer

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
    parser = argparse.ArgumentParser(description="Train, or apply a speech classifier.")
    parser.add_argument("--mode", choices=['train', 'apply'], required=True, help="Mode of operation: 'train', 'evaluate', or 'apply'.")
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

        X_features, y_labels, group_ids = prepare_lgbm_classifier_data(rttm_df, AUDIO_DIR)

        trained_model = None
        trained_imputer = None 

        if X_features is not None and y_labels is not None and group_ids is not None and len(X_features) > 0:
            # 2. Train the classifier with LightGBM
            trained_model, trained_imputer = train_lgbm_classifier(X_features, y_labels, group_ids)
        else:
            print("Skipping LGBM model training due to lack of data or group_ids.")
        
        if trained_model and trained_imputer:
            print("\nClassifier training complete.")
            joblib.dump(trained_model, MODEL_SAVE_PATH)
            joblib.dump(trained_imputer, IMPUTER_SAVE_PATH)
            print(f"Trained model saved to {MODEL_SAVE_PATH}")
            print(f"Trained imputer saved to {IMPUTER_SAVE_PATH}")
        else:
            print("\nClassifier training failed or was skipped.")

    elif args.mode == 'apply':
        print("--- Running in APPLY mode ---")
        if not args.input_rttm_apply or not os.path.exists(args.input_rttm_apply):
            print(f"Error: Input RTTM file for apply mode not found at {args.input_rttm_apply}")
            exit()

        rttm_content = read_rttm_file(args.input_rttm_apply)
        if rttm_content is None:
            print("No RTTM data available for applying the model. Exiting.")
            exit()

        rttm_df = parse_rttm(rttm_content)
        if rttm_df.empty:
            print("Parsed RTTM is empty for applying the model. No data to process. Check RTTM content and EXPECTED_LABELS.")
            exit()

        print(f"\nParsed RTTM data for applying the model ({len(rttm_df)} segments with expected labels)")

        # Load the trained model and imputer
        if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(IMPUTER_SAVE_PATH):
            trained_model = joblib.load(MODEL_SAVE_PATH)
            trained_imputer = joblib.load(IMPUTER_SAVE_PATH)
            print("Loaded trained model and imputer successfully.")
        else:
            print("Trained model or imputer not found. Cannot apply classification.")
            exit()

        final_classifications_df = apply_full_pipeline(rttm_df, AUDIO_DIR, trained_model, trained_imputer)

        if final_classifications_df is not None:
            output_csv_path = args.output_csv_apply or OUTPUT_CSV_PATH
            final_classifications_df.to_csv(output_csv_path, index=False)
            print(f"Final classifications saved to {output_csv_path}")
        else:
            print("No classifications were made. Exiting.")