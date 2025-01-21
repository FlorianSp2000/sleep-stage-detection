import pandas as pd
import mne
import yasa
import pickle

def load_and_preprocess_labels(file_paths, offset=30):
    """
    Load and preprocess label files.
    Parameters:
        file_paths (list): List of file paths to process.
        offset (int): Number of entries to exclude from the end of the series.
    Returns:
        pd.Series: Concatenated labels after preprocessing.
    """
    labels_list = []
    for path in file_paths:
        labels = pd.read_csv(path, header=None).squeeze("columns")
        labels.loc[labels == 5] = 4
        labels = labels[:-offset]
        labels_list.append(labels)
    return pd.concat(labels_list, axis=0, ignore_index=True)

def compute_label_agreement(labels1, labels2):
    total_labels = len(labels1)
    matches = labels1 == labels2
    num_matches = sum(matches)
    num_differences = total_labels - num_matches
    agreement_percentage = (num_matches / total_labels) * 100
    
    classes = sorted(set(labels1) | set(labels2))
    class_metrics = {cls: {'total': 0, 'matches': 0} for cls in classes}
    
    for l1, l2 in zip(labels1, labels2):
        if l1 == l2:
            class_metrics[l1]['total'] += 1
            class_metrics[l1]['matches'] += 1
        else:
            class_metrics[l1]['total'] += 1
            class_metrics[l2]['total'] += 1
    
    for cls in class_metrics:
        total = class_metrics[cls]['total']
        matches = class_metrics[cls]['matches']
        class_metrics[cls]['agreement_percentage'] = (matches / total * 100) if total > 0 else 0
    
    return {
        "total_labels": total_labels,
        "num_matches": num_matches,
        "num_differences": num_differences,
        "agreement_percentage": agreement_percentage,
        "class_metrics": class_metrics
    }

def load_and_preprocess_edf(file_paths, channels_to_drop, resample_rate=100, filter_range=(0.3, 49), eeg_name='C4-A1', eog_name='LOC-A2', emg_name='X1'):
    """
    Load and preprocess EDF files.
    Parameters:
        file_paths (list): List of EDF file paths to process.
        channels_to_drop (list): Channels to drop from the raw data.
        resample_rate (int): Resampling rate for the data.
        filter_range (tuple): Frequency range for filtering (low, high).
        eeg_name, eog_name, emg_name (str): Channel names for sleep staging.
    Returns:
        list: Preprocessed feature DataFrames for each file.
    """
    data_list = []
    for i, path in enumerate(file_paths, start=1):
        raw = mne.io.read_raw_edf(path, preload=True)
        raw.drop_channels(channels_to_drop)
        raw.resample(resample_rate)
        raw.filter(*filter_range)
        sls = yasa.SleepStaging(raw, eeg_name=eeg_name, eog_name=eog_name, emg_name=emg_name)
        features = sls.get_features()[:-30]
        features['patient'] = i
        data_list.append(features)
    return data_list

def build_train_test_set():
    """
    Build and save the train-test split for the dataset as pickle file.
    """
    # File paths for labels
    label_paths1 = [f"../dataset/{i}/{i}_1.txt" for i in range(1, 11)]
    label_paths2 = [f"../dataset/{i}/{i}_2.txt" for i in range(1, 11)]

    # Load and preprocess labels
    labels1 = load_and_preprocess_labels(label_paths1)
    labels2 = load_and_preprocess_labels(label_paths2)

    # File paths for EDF data
    edf_paths = [f"../dataset/{i}/{i}.edf" for i in range(1, 11)]

    # Channels to drop
    channels_to_drop = ['X4', 'X5', 'X6', 'DC3', 'X7', 'X8', 'SaO2', 'DC8', 'ROC-A1', 
                        'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'O2-A1', 'X2', 'X3']

    # Load and preprocess EDF data
    edf_features = load_and_preprocess_edf(
        edf_paths, channels_to_drop, eeg_name='C4-A1', eog_name='LOC-A2', emg_name='X1'
    )

    df = pd.concat(edf_features, axis=0, ignore_index=True)
    df['label'] = labels1
    
    df_train = df.loc[df['patient'].between(1, 8)].copy()
    df_test = df.loc[df['patient'].between(9, 10)].copy()

    y_train = df_train['label'].copy()
    X_train = df_train.drop(columns=['label']).copy()

    y_test = df_test['label'].copy()
    X_test = df_test.drop(columns=['label']).copy()

    assert set(df_train['patient']).isdisjoint(set(df_test['patient'])), "Train and test sets overlap!"
    assert len(df_train) + len(df_test) == len(df), "Train and test sets do not cover the full dataset!"

    split_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }
    with open('../dataset/train_test_split.pkl', 'wb') as f:
        pickle.dump(split_data, f)

    return X_train, y_train, X_test, y_test

def load_train_test_set():
    with open('../dataset/train_test_split.pkl', 'rb') as f:
        split_data = pickle.load(f)
    return split_data['X_train'], split_data['y_train'], split_data['X_test'], split_data['y_test']