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
import soundfile as sf # For dummy audio creation

# --- Configuration ---
AUDIO_DIR = "/home/nele_pauline_suffo/ProcessedData/childlens_audio"
TRAIN_RTTM_FILE_PATH = "/home/nele_pauline_suffo/ProcessedData/vtc_childlens_v2/train.rttm"

RTTM_COLUMNS = ['type', 'file_id', 'channel', 'start_time', 'duration', 
                'NA1', 'NA2', 'diarization_label', 'NA3', 'NA4']
EXPECTED_LABELS = ["KCHI", "OHS", "CDS"] # Labels to be learned by the classifier

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
            
            # We will handle NaNs with an imputer before training
            # but we should only include segments for which we could extract some features
            # and have a valid label.
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
def train_speech_classifier(X, y):
    """
    Trains a classifier for KCHI vs. OHS vs. CDS.
    X: Feature vectors.
    y: True labels ('KCHI', 'OHS', 'CDS').
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

    # Stratify by y to ensure proportional representation of classes in train/test splits
    # Test size can be adjusted, e.g., 0.2 for 20% test data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.25, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Error during train_test_split (likely due to insufficient samples for a class for stratification): {e}")
        print(f"Class distribution in y: {pd.Series(y).value_counts().to_dict()}")
        # Fallback: don't stratify if it fails, though this is not ideal
        if len(np.unique(y)) == 1 and len(y) > 1: # If only one class, can't stratify or split meaningfully for classification
             print("Only one class present in the data. Cannot perform train/test split for classification evaluation.")
             # Optionally, train on all data if only one class, but evaluation will be trivial
             X_train, X_test, y_train, y_test = X_imputed, np.array([]), y, np.array([])
        elif len(y) > 1 : # if more than one sample, try without stratify
            print("Attempting train_test_split without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_imputed, y, test_size=0.25, random_state=42
            )
        else: # Not enough data to split
            print("Not enough data to perform train/test split.")
            return None, None


    
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

# --- (apply_full_pipeline function is commented out as its original purpose is superseded) ---
# If you need to apply the trained 3-class model to new data later,
# you would write a new function similar to this, but it would take
# the 3-class model and apply it.
"""
def apply_full_pipeline(rttm_df, audio_base_path, trained_speech_model, feature_imputer):
    final_classifications = []
    if trained_speech_model is None or feature_imputer is None:
        print("Trained model or imputer not available. Cannot perform full classification.")
        # Fallback or error handling
        for index, row in rttm_df.iterrows():
            final_classifications.append({
                **row.to_dict(),
                'final_classification': 'MODEL_MISSING'
            })
        return pd.DataFrame(final_classifications)

    for index, row in rttm_df.iterrows():
        # Assuming rttm_df here is for new data that needs classification
        # and might only have basic diarization labels (e.g. SPEAKER_01, SPEAKER_02)
        # or KCHI/FEM/MAL which then need to be mapped by this model.
        # For this example, let's assume we are re-classifying based on features.
        
        segment_audio_path = os.path.join(audio_base_path, row['file_id'] + ".wav")
        classification = "ERROR_UNKNOWN"
        if not os.path.exists(segment_audio_path):
            classification = "ERROR_NO_AUDIO"
        else:
            y, sr = load_audio_segment(segment_audio_path, row['start_time'], row['duration'])
            if y is None:
                classification = "ERROR_LOAD_AUDIO"
            else:
                features = extract_prosodic_features(y, sr)
                features_reshaped = np.array(features).reshape(1, -1)
                features_imputed = feature_imputer.transform(features_reshaped)
                
                if any(np.isnan(features_imputed.flatten())):
                    classification = "ERROR_FEATURE_IMPUTATION_FAILED"
                else:
                    classification = trained_speech_model.predict(features_imputed)[0]
        
        final_classifications.append({
            'file_id': row['file_id'],
            'start_time': row['start_time'],
            'duration': row['duration'],
            'original_diarization_label': row.get('diarization_label', 'NA'), # if present
            'predicted_speech_class': classification
        })
    return pd.DataFrame(final_classifications)
"""

def read_rttm_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: RTTM file not found at {file_path}")
        return None

if __name__ == "__main__":
    rttm_content = None
    if os.path.exists(TRAIN_RTTM_FILE_PATH):
        print(f"Loading RTTM from: {TRAIN_RTTM_FILE_PATH}")
        rttm_content = read_rttm_file(TRAIN_RTTM_FILE_PATH)
    else:
        print(f"Warning: {TRAIN_RTTM_FILE_PATH} not found.")

    if rttm_content is None:
        print("No RTTM data available. Exiting.")
        exit()

    rttm_df = parse_rttm(rttm_content)
    if rttm_df.empty:
        print("Parsed RTTM is empty. No data to process. Check RTTM content and EXPECTED_LABELS.")
        exit()
    
    print(f"\nParsed RTTM data ({len(rttm_df)} segments)")

    # 1. Prepare data for classifier training
    X_features, y_labels, processed_segments = prepare_classifier_data(rttm_df, AUDIO_DIR)

    trained_model = None
    trained_imputer = None 

    if X_features is not None and y_labels is not None and len(X_features) > 0 :
        # 2. Train the speech classifier
        print(f"\n--- Training Speech Classifier (KCHI, OHS, CDS) ---")
        trained_model, trained_imputer = train_speech_classifier(X_features, y_labels)
        if trained_model:
            print("\nClassifier training complete.")
            # You can save the trained_model and trained_imputer here using joblib or pickle
            import joblib
            joblib.dump(trained_model, 'speech_classifier_pipeline.pkl')
            joblib.dump(trained_imputer, 'speech_feature_imputer.pkl')
            print("Trained model and imputer saved.")
        else:
            print("\nClassifier training failed or was skipped.")
    else:
        print("\nSkipping classifier training due to lack of data or errors during preparation.")

    # The `apply_full_pipeline` logic would go here if you wanted to apply the
    # trained model to new/unlabeled data. For now, the script focuses on training
    # and evaluating the 3-class classifier from your pre-labeled RTTM.
    print("\n--- Script Finished ---")