import mne
from mne.datasets import sample
from pathlib import Path # Import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

def load_sample_data():
    """
    Loads the MNE sample dataset (auditory/visual stimuli).
    Downloads it if not already present.
    """
    # Get the root path of the MNE sample dataset. This will trigger download if not present.
    dataset_root_path = sample.data_path(verbose=True)
    
    # Construct the path to the specific raw file using pathlib
    raw_fname = Path(dataset_root_path) / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
    
    # basic error handling
    if not raw_fname.exists():
        # This case should be less common if sample.data_path() succeeded and the dataset is intact.
        # If sample.data_path() itself failed, it would usually raise an error.
        print(f"Raw file not found at {raw_fname}. The MNE sample dataset might be incomplete or corrupted.")
        print("Try removing the directory '~/mne_data/MNE-sample-data' and running the script again to force a fresh download.")
        raise FileNotFoundError(f"Failed to find {raw_fname} within the MNE sample dataset.")

    # load the data
    print(f"Loading raw data from: {raw_fname}")
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    print("Data loaded successfully.")
    
    # print some basic info
    print(raw) # print the raw object
    print(raw.info) # print the info object which contains the channel names, bad channels, etc.
    
    return raw

def find_and_examine_events(raw):
    """
    Find events in the raw data and examine what types of stimuli were presented.
    Events are markers that indicate when stimuli occurred.
    """
    print("\n" + "="*50)
    print("STEP 1: Finding and examining events")
    print("="*50)
    
    # Find events in the data
    # Events are stored in stimulus channels (STI channels)
    events = mne.find_events(raw, stim_channel='STI 014', verbose=True)
    
    print(f"\nFound {len(events)} events in the data.")
    print(f"Events array shape: {events.shape}")
    print("\nFirst 10 events:")
    print("Sample\tTime(s)\tEvent_ID")
    print("-" * 30)
    for i in range(min(10, len(events))):
        sample_num = events[i, 0]
        time_sec = sample_num / raw.info['sfreq']  # Convert sample to time
        event_id = events[i, 2]
        print(f"{sample_num}\t{time_sec:.2f}\t{event_id}")
    
    # Get unique event IDs and their counts
    unique_events, counts = np.unique(events[:, 2], return_counts=True)
    
    print(f"\nUnique event IDs and their counts:")
    print("Event_ID\tCount\tDescription")
    print("-" * 40)
    
    # The MNE sample dataset has specific event IDs:
    # 1 = Auditory/Left, 2 = Auditory/Right, 3 = Visual/Left, 4 = Visual/Right
    # 5 = Smiley face, 32 = Button press
    event_descriptions = {
        1: "Auditory/Left",
        2: "Auditory/Right", 
        3: "Visual/Left",
        4: "Visual/Right",
        5: "Smiley face",
        32: "Button press"
    }
    
    for event_id, count in zip(unique_events, counts):
        description = event_descriptions.get(event_id, "Unknown")
        print(f"{event_id}\t\t{count}\t{description}")
    
    # For our binary classification task (Auditory vs Visual), we'll group:
    # Auditory: event IDs 1, 2 (left and right auditory)
    # Visual: event IDs 3, 4 (left and right visual)
    
    auditory_events = events[np.isin(events[:, 2], [1, 2])]
    visual_events = events[np.isin(events[:, 2], [3, 4])]
    
    print(f"\nFor binary classification (Auditory vs Visual):")
    print(f"Auditory events (IDs 1,2): {len(auditory_events)}")
    print(f"Visual events (IDs 3,4): {len(visual_events)}")
    print(f"Total usable events: {len(auditory_events) + len(visual_events)}")
    
    # Plot the events to visualize their timing
    print(f"\nPlotting events timeline...")
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot different event types with different colors
    for event_id in unique_events:
        if event_id in [1, 2, 3, 4]:  # Only plot the main stimulus events
            event_times = events[events[:, 2] == event_id, 0] / raw.info['sfreq']
            color = 'red' if event_id in [1, 2] else 'blue'
            label = f"Auditory ({event_descriptions[event_id]})" if event_id in [1, 2] else f"Visual ({event_descriptions[event_id]})"
            ax.scatter(event_times, [event_id] * len(event_times), 
                      c=color, alpha=0.7, s=30, label=label)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Event ID')
    ax.set_title('Event Timeline: Auditory (Red) vs Visual (Blue) Stimuli')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return events

def create_epochs(raw, events):
    """
    STEP 2: Create epochs around events.
    Epochs are time-locked segments of data around each stimulus.
    """
    print("\n" + "="*50)
    print("STEP 2: Creating epochs")
    print("="*50)
    
    # Define event IDs for our binary classification
    event_id = {
        'auditory/left': 1,
        'auditory/right': 2,
        'visual/left': 3,
        'visual/right': 4
    }
    
    # Define time window around events
    # -0.2 to 0.5 seconds around stimulus onset is common for MEG
    tmin, tmax = -0.2, 0.5  # seconds
    
    print(f"Creating epochs from {tmin} to {tmax} seconds around stimulus onset...")
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                       baseline=(None, 0),  # Baseline correction from start to stimulus onset
                       reject=dict(grad=4000e-13, mag=4e-12),  # Reject noisy epochs >4000 fT/cm and >4 pT
                       preload=True, verbose=True)
    
    print(f"\nEpochs created successfully!")
    print(f"Epochs shape: {epochs.get_data().shape}")  # (n_epochs, n_channels, n_times)
    print(f"Number of epochs: {len(epochs)}")
    print(f"Number of channels: {epochs.info['nchan']}")
    print(f"Number of time points per epoch: {len(epochs.times)}")
    print(f"Time range: {epochs.times[0]:.3f} to {epochs.times[-1]:.3f} seconds")
    
    # Show epochs by condition
    print(f"\nEpochs by condition:")
    for condition in event_id.keys():
        n_epochs = len(epochs[condition])
        print(f"  {condition}: {n_epochs} epochs")
    
    # Plot some example epochs
    print(f"\nPlotting example epochs...")
    
    # Pick a few MEG channels for visualization
    picks = mne.pick_types(epochs.info, meg='grad', exclude='bads')[:6]  # First 6 good gradiometers
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, pick in enumerate(picks):
        ch_name = epochs.info['ch_names'][pick]
        
        # Plot average response for auditory and visual conditions
        auditory_data = epochs['auditory/left', 'auditory/right'].get_data()[:, pick, :].mean(axis=0)
        visual_data = epochs['visual/left', 'visual/right'].get_data()[:, pick, :].mean(axis=0)
        
        axes[i].plot(epochs.times, auditory_data * 1e13, 'r-', label='Auditory', linewidth=2)
        axes[i].plot(epochs.times, visual_data * 1e13, 'b-', label='Visual', linewidth=2)
        axes[i].axvline(0, color='k', linestyle='--', alpha=0.5, label='Stimulus onset')
        axes[i].set_title(f'{ch_name}')
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (fT/cm)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return epochs

def extract_features(epochs):
    """
    STEP 3: Extract features from epochs for machine learning.
    We'll extract multiple types of features commonly used in MEG analysis.
    """
    print("\n" + "="*50)
    print("STEP 3: Feature extraction")
    print("="*50)
    
    # Get the epoch data
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    print(f"Extracting features from {n_epochs} epochs...")
    print(f"Data shape: {data.shape}")
    
    # Only use MEG channels (exclude EEG, EOG, etc.)
    meg_picks = mne.pick_types(epochs.info, meg=True, exclude='bads')
    meg_data = data[:, meg_picks, :]
    n_meg_channels = len(meg_picks)
    
    print(f"Using {n_meg_channels} MEG channels for feature extraction")
    
    # Initialize feature matrix
    features_list = []
    feature_names = []
    
    # 1. TEMPORAL FEATURES
    print("Extracting temporal features...")
    
    # Mean amplitude in different time windows
    time_windows = [
        (0.0, 0.1, "early"),      # Early response (0-100ms)
        (0.1, 0.3, "middle"),     # Middle response (100-300ms)
        (0.3, 0.5, "late")        # Late response (300-500ms)
    ]
    
    for t_start, t_end, window_name in time_windows:
        # Find time indices
        time_mask = (epochs.times >= t_start) & (epochs.times <= t_end)
        
        # Mean amplitude in this time window for each channel
        window_mean = meg_data[:, :, time_mask].mean(axis=2)  # Shape: (n_epochs, n_channels)
        features_list.append(window_mean)
        
        # Add feature names
        for ch_idx in range(n_meg_channels):
            feature_names.append(f"mean_amp_{window_name}_ch{ch_idx}")
    
    # Peak amplitude and latency
    print("Extracting peak features...")
    
    # Find peak amplitude in post-stimulus period (0-500ms)
    post_stim_mask = epochs.times >= 0
    post_stim_data = meg_data[:, :, post_stim_mask]
    
    # Peak amplitude (absolute value)
    peak_amp = np.max(np.abs(post_stim_data), axis=2)
    features_list.append(peak_amp)
    for ch_idx in range(n_meg_channels):
        feature_names.append(f"peak_amp_ch{ch_idx}")
    
    # 2. FREQUENCY FEATURES
    print("Extracting frequency features...")
    
    # Define frequency bands
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    sfreq = epochs.info['sfreq']
    
    for band_name, (low_freq, high_freq) in freq_bands.items():
        print(f"  Computing {band_name} band power ({low_freq}-{high_freq} Hz)...")
        
        band_power = np.zeros((n_epochs, n_meg_channels))
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_meg_channels):
                # Compute power spectral density using Welch's method
                freqs, psd = signal.welch(meg_data[epoch_idx, ch_idx, :], 
                                        sfreq, nperseg=min(256, n_times))
                
                # Find frequency indices for this band
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                # Average power in this frequency band
                band_power[epoch_idx, ch_idx] = np.mean(psd[freq_mask])
        
        features_list.append(band_power)
        for ch_idx in range(n_meg_channels):
            feature_names.append(f"{band_name}_power_ch{ch_idx}")
    
    # 3. SPATIAL FEATURES
    print("Extracting spatial features...")
    
    # Global Field Power (GFP) - measure of overall activity
    gfp = np.std(meg_data, axis=1)  # Shape: (n_epochs, n_times)
    
    # Mean GFP in different time windows
    for t_start, t_end, window_name in time_windows:
        time_mask = (epochs.times >= t_start) & (epochs.times <= t_end)
        gfp_mean = gfp[:, time_mask].mean(axis=1)  # Shape: (n_epochs,)
        features_list.append(gfp_mean.reshape(-1, 1))
        feature_names.append(f"gfp_mean_{window_name}")
    
    # Peak GFP
    gfp_peak = np.max(gfp[:, post_stim_mask], axis=1)
    features_list.append(gfp_peak.reshape(-1, 1))
    feature_names.append("gfp_peak")
    
    # Combine all features
    X = np.concatenate(features_list, axis=1)
    
    print(f"\nFeature extraction complete!")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of epochs: {X.shape[0]}")
    
    # Create labels for binary classification
    # Combine left and right conditions into auditory vs visual
    labels = []
    for i in range(len(epochs)):
        event_id = epochs.events[i, 2]  # Get event ID from events array
        if event_id in [1, 2]:  # Auditory
            labels.append(0)
        elif event_id in [3, 4]:  # Visual
            labels.append(1)
    
    y = np.array(labels)
    
    print(f"\nLabels created:")
    print(f"Auditory (0): {np.sum(y == 0)} epochs")
    print(f"Visual (1): {np.sum(y == 1)} epochs")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(f"Mean feature value: {X.mean():.2e}")
    print(f"Std feature value: {X.std():.2e}")
    print(f"Min feature value: {X.min():.2e}")
    print(f"Max feature value: {X.max():.2e}")
    
    return X, y, feature_names

def train_and_evaluate_models(X, y, feature_names):
    """
    STEP 4: Train and evaluate machine learning models.
    We'll test multiple algorithms and use cross-validation for robust evaluation.
    """
    print("\n" + "="*50)
    print("STEP 4: Machine Learning Models")
    print("="*50)
    
    print(f"Training models on {X.shape[0]} samples with {X.shape[1]} features...")
    print(f"Class distribution: {np.bincount(y)} (Auditory: {np.sum(y==0)}, Visual: {np.sum(y==1)})")
    
    # 1. DATA PREPROCESSING
    print("\n1. Data preprocessing...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Standardize features (important for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized (mean=0, std=1)")
    
    # 2. DEFINE MODELS
    print("\n2. Defining models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM (RBF)': SVC(random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # 3. CROSS-VALIDATION
    print("\n3. Cross-validation evaluation...")
    
    # Use stratified k-fold to maintain class balance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Use scaled data for SVM and Logistic Regression, original for Random Forest
        if name in ['Logistic Regression', 'SVM (RBF)']:
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        else:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        
        cv_results[name] = scores
        
        print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"  Individual folds: {[f'{s:.3f}' for s in scores]}")
    
    # 4. TRAIN FINAL MODELS AND TEST
    print("\n4. Final model training and testing...")
    
    test_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining final {name} model...")
        
        # Train on full training set
        if name in ['Logistic Regression', 'SVM (RBF)']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate test accuracy
        test_acc = accuracy_score(y_test, y_pred)
        test_results[name] = {
            'accuracy': test_acc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        trained_models[name] = model
        
        print(f"  Test Accuracy: {test_acc:.3f}")
        
        # Detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['Auditory', 'Visual']))
    
    # 5. VISUALIZE RESULTS
    print("\n5. Visualizing results...")
    
    # Plot cross-validation results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # CV accuracy comparison
    model_names = list(cv_results.keys())
    cv_means = [cv_results[name].mean() for name in model_names]
    cv_stds = [cv_results[name].std() for name in model_names]
    
    axes[0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
    axes[0].set_title('Cross-Validation Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add chance level line
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance (50%)')
    axes[0].legend()
    
    # Test accuracy comparison
    test_accs = [test_results[name]['accuracy'] for name in model_names]
    axes[1].bar(model_names, test_accs, alpha=0.7, color='orange')
    axes[1].set_title('Test Set Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance (50%)')
    axes[1].legend()
    
    # Confusion matrix for best model
    best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['accuracy'])
    best_predictions = test_results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Auditory', 'Visual'],
                yticklabels=['Auditory', 'Visual'], ax=axes[2])
    axes[2].set_title(f'Confusion Matrix - {best_model_name}')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()
    
    # 6. FEATURE IMPORTANCE (for Random Forest)
    if 'Random Forest' in trained_models:
        print("\n6. Feature importance analysis...")
        
        rf_model = trained_models['Random Forest']
        feature_importance = rf_model.feature_importances_
        
        # Get top 20 most important features
        top_indices = np.argsort(feature_importance)[-20:]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = feature_importance[top_indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.show()
        
        print("Top 10 most important features:")
        for i, (feat, imp) in enumerate(zip(top_features[-10:], top_importance[-10:])):
            print(f"  {i+1:2d}. {feat}: {imp:.4f}")
    
    # 7. SUMMARY
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"Best performing model: {best_model_name}")
    print(f"Best test accuracy: {test_results[best_model_name]['accuracy']:.3f}")
    
    print(f"\nAll model performances:")
    for name in model_names:
        cv_acc = cv_results[name].mean()
        test_acc = test_results[name]['accuracy']
        print(f"  {name:20s}: CV = {cv_acc:.3f} ± {cv_results[name].std():.3f}, Test = {test_acc:.3f}")
    
    # Check if results are above chance
    chance_level = 0.5
    best_acc = test_results[best_model_name]['accuracy']
    
    if best_acc > chance_level + 0.1:  # 10% above chance
        print(f"\n✅ SUCCESS: Models can distinguish auditory from visual brain responses!")
        print(f"   Best accuracy ({best_acc:.1%}) is well above chance level ({chance_level:.1%})")
    elif best_acc > chance_level + 0.05:  # 5% above chance
        print(f"\n⚠️  MODERATE: Some ability to distinguish auditory from visual responses")
        print(f"   Best accuracy ({best_acc:.1%}) is moderately above chance level ({chance_level:.1%})")
    else:
        print(f"\n❌ POOR: Models struggle to distinguish auditory from visual responses")
        print(f"   Best accuracy ({best_acc:.1%}) is close to chance level ({chance_level:.1%})")
    
    return trained_models, test_results, cv_results

if __name__ == '__main__':
    # Load the data
    raw_data = load_sample_data()
    
    # Step 1: Find and examine events
    events = find_and_examine_events(raw_data)
    
    # Step 2: Create epochs
    epochs = create_epochs(raw_data, events)
    
    # Step 3: Extract features
    X, y, feature_names = extract_features(epochs)
    
    # Step 4: Train and evaluate machine learning models
    models, test_results, cv_results = train_and_evaluate_models(X, y, feature_names)
    
    print("\n🎉 MEG Machine Learning Pipeline Complete!")
    print("You've successfully built a classifier to distinguish auditory from visual brain responses!")
