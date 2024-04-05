import os
import librosa
import opensmile
import pandas as pd
from tqdm import tqdm
from audio_io import load_audio_file, save_features, load_segmentation_file

# Hardcoded window sizes for analysis
window_sizes = [
    {"n_fft": 2048, "hop_length": 512},
    # ...
]


def process_segment(segment, sr, smile, window_config):
    """
    Process a single audio segment to extract features, considering the window configuration.

    Parameters:
    - segment: The audio segment.
    - sr: Sample rate of the audio segment.
    - smile: Configured opensmile instance.
    - window_config: Dict specifying 'n_fft' and 'hop_length'.

    Returns:
    - Features extracted from the segment.
    """
    features = smile.process_signal(segment, sr)
    return features


def extract_egemaps_features(audio_path, segments_df, smile, window_config):
    """
    Extracts eGeMAPS features from specific segments of an audio file using opensmile,
    for each window configuration.

    Parameters:
    - audio_path: Path to the audio file.
    - segments_df: DataFrame containing segmentation data.
    - smile: Configured opensmile instance.
    - window_config: Dict specifying 'n_fft' and 'hop_length'.

    Returns:
    - A DataFrame containing the extracted eGeMAPS features for specified segments.
    """
    all_features = []

    # Process each participant segment
    for _, segment in segments_df.iterrows():
        y, sr = librosa.load(audio_path, sr=None, offset=segment['begin'], duration=segment['end'] - segment['begin'])
        segment_features = process_segment(y, sr, smile, window_config)
        all_features.append(segment_features)

    # Combine all features into a single DataFrame
    if all_features:
        return pd.concat(all_features, ignore_index=True)
    else:
        return pd.DataFrame()


def preprocess_audio(partition):
    print(f"Processing partition: {partition}")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPS,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    audio_dir = os.path.join('data', 'raw', 'audio', partition)
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

    for window_config in window_sizes:
        for filename in tqdm(audio_files, desc=f"Window config: {window_config}"):
            audio_path, label = load_audio_file(partition, filename)
            if audio_path is None:
                continue

            base_name = os.path.splitext(filename)[0]
            segments_df = load_segmentation_file(partition, base_name)
            if segments_df is None:
                continue

            par_segments = segments_df[segments_df['speaker'] == 'PAR']
            features_df = extract_egemaps_features(audio_path, par_segments, smile, window_config)
            features_df['Label'] = label

            save_features(partition, base_name, features_df, window_config)


if __name__ == "__main__":
    for partition in ['ad', 'cn']:
        preprocess_audio(partition)