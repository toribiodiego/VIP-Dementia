import os
import pandas as pd

HOME_DIRECTORY = "/Users/Work/PycharmProjects/VIP/pipeline"

RAW_DATA_DIR = os.path.join(HOME_DIRECTORY, 'data/raw')
PROCESSED_DATA_DIR = os.path.join(HOME_DIRECTORY, 'data/processed')
FEATURES_DIR = os.path.join(PROCESSED_DATA_DIR, 'features')
LABELS_DIR = os.path.join(PROCESSED_DATA_DIR, 'labels')


def get_sample_filename(partition, filename):
    """Constructs the full path to a sample given its partition and filename."""
    directory = 'ad' if partition == 'ad' else 'cn'
    return os.path.join(RAW_DATA_DIR, 'audio', directory, filename)



def get_feature_filename(partition, index):
    """Constructs a filename for a feature file given its partition and index."""
    return f"{partition}_{str(index).zfill(3)}_features.csv"


def load_audio_file(partition, filename):
    """Loads an audio file given its partition and filename."""
    directory = 'ad' if partition == 'ad' else 'cn'
    filepath = os.path.join(RAW_DATA_DIR, 'audio', directory, filename)
    print(f"audio file at: {filepath}")  # Print the path being checked
    label = 'AD' if partition == 'ad' else 'CN'
    if not os.path.exists(filepath):
        print(f"Audio file {filename} not found in {directory} directory.")
        return None, None
    return filepath, label



def load_labels(partition):
    """Loads labels from a CSV file for a given partition."""
    labels_path = os.path.join(LABELS_DIR, f"{partition}_labels.csv")
    if not os.path.exists(labels_path):
        print(f"Label file for {partition} not found.")
        return None
    return pd.read_csv(labels_path)


def load_features(partition, index):
    """Loads specific features for a given partition and index."""
    directory = 'ad' if partition == 'ad' else 'cn'
    filename = get_feature_filename(partition, index)
    filepath = os.path.join(FEATURES_DIR, directory, filename)
    if not os.path.exists(filepath):
        print(f"Feature file {filename} for {partition} not found.")
        return None
    return pd.read_csv(filepath)


def save_features(partition, base_name, features, window_config):
    """
    Saves features to a CSV file for a given partition and base name, considering window configuration.
    """
    directory = 'ad' if partition == 'ad' else 'cn'
    feature_dir = os.path.join(FEATURES_DIR, directory)
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)
    window_details = f"{window_config['n_fft']}_{window_config['hop_length']}"
    filename = f"{base_name}_{window_details}_features.csv"
    filepath = os.path.join(feature_dir, filename)
    features.to_csv(filepath, index=False)
    print(f"Features saved to {filepath} using window config: {window_config}")


def save_labels(partition, labels):
    """Saves labels to a CSV file for a given partition."""
    labels_path = os.path.join(LABELS_DIR, f"{partition}_labels.csv")
    labels.to_csv(labels_path, index=False)
    print(f"Labels saved to {labels_path}")


def load_segmentation_file(partition, base_name):
    """Loads the segmentation file for a given partition and base filename."""
    directory = 'ad' if partition == 'ad' else 'cn'
    segmentation_path = os.path.join(RAW_DATA_DIR, 'segmentation', f"{base_name}.csv")

    print(f"Attempting to load segmentation file at: {segmentation_path}")  # Debugging print statement

    if not os.path.exists(segmentation_path):
        print(f"Segmentation file {base_name}.csv not found in {directory} directory.")
        return None

    return pd.read_csv(segmentation_path)



