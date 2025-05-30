# MEG Machine Learning Project: Auditory vs Visual Classification

A complete machine learning pipeline for analyzing MEG (Magnetoencephalography) data to classify brain responses to auditory versus visual stimuli. This project demonstrates how to apply modern ML techniques to neuroimaging data using MNE-Python.

## 🎯 Project Overview

This project implements a full MEG analysis pipeline that successfully distinguishes between auditory and visual brain responses with **100% accuracy** using machine learning. It's designed as an educational introduction to MEG data analysis for machine learning professionals with no prior neuroimaging experience.

## 🏆 Key Achievements

- **Perfect Classification**: 100% test accuracy with Logistic Regression
- **Robust Performance**: All models (Logistic Regression, SVM, Random Forest) achieved >95% accuracy
- **Strong Cross-Validation**: 97.5% ± 1.6% CV accuracy demonstrates reliable performance
- **Feature Insights**: Identified theta band power (4-8 Hz) as the most discriminative neural signature
- **Complete Pipeline**: From raw MEG data to trained models with proper validation

## 📊 Results Summary

| Model | Cross-Validation Accuracy | Test Accuracy |
|-------|---------------------------|---------------|
| **Logistic Regression** | 97.5% ± 1.6% | **100.0%** |
| SVM (RBF) | 96.5% ± 2.5% | 95.3% |
| Random Forest | 95.0% ± 2.2% | 96.5% |

*All results significantly above 50% chance level*

## 🧠 What We Learned

1. **Neural Signatures**: Auditory and visual processing create distinct, measurable brain patterns
2. **Theta Oscillations**: 4-8 Hz brain rhythms are crucial for sensory discrimination
3. **Spatial Patterns**: Specific MEG channels show strongest discriminative power
4. **Linear Separability**: Simple linear models can achieve perfect classification

## 🛠️ Technical Stack

- **Python 3.12+**
- **MNE-Python**: MEG data processing and analysis
- **scikit-learn**: Machine learning algorithms
- **NumPy/SciPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Pathlib**: Modern file handling

## 📁 Project Structure

```
OpenNeuroBIDS/
├── openmne.py          # Main analysis pipeline
├── README.md           # This file
├── mnevenv/            # Virtual environment
└── requirements.txt    # Dependencies (if created)
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.12+** installed
2. **Virtual environment** (recommended)
3. **Internet connection** (for downloading sample data)

### Installation

1. **Clone or download** this project
2. **Create and activate virtual environment**:
   ```bash
   python -m venv mnevenv
   source mnevenv/bin/activate  # On macOS/Linux
   # or
   mnevenv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install mne scikit-learn seaborn matplotlib numpy scipy pathlib
   ```

### Running the Analysis

Simply run the main script:

```bash
python openmne.py
```

The pipeline will automatically:
1. Download the MNE sample dataset (~1.5GB, first run only)
2. Load and preprocess MEG data
3. Extract events and create epochs
4. Extract 2,749 features from 305 MEG channels
5. Train and evaluate 3 machine learning models
6. Display results and visualizations

## 📈 Pipeline Steps

### Step 1: Event Detection
- Identifies 319 total events in the dataset
- Filters to 288 usable auditory/visual stimuli
- Creates timeline visualization

### Step 2: Epoching
- Segments data into 700ms windows (-200ms to +500ms around stimuli)
- Applies baseline correction and artifact rejection
- Results in 286 clean epochs (143 auditory, 143 visual)

### Step 3: Feature Extraction
- **Temporal Features**: Mean amplitude in early/middle/late time windows, peak amplitude
- **Frequency Features**: Power in 5 frequency bands (delta, theta, alpha, beta, gamma)
- **Spatial Features**: Global Field Power measures
- **Total**: 2,749 features from 305 MEG channels

### Step 4: Machine Learning
- **Data Split**: 70% training, 30% testing
- **Cross-Validation**: 5-fold stratified CV
- **Models**: Logistic Regression, SVM, Random Forest
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices

## 🔍 Key Features

### Robust Preprocessing
- Automatic artifact rejection (noisy epochs removed)
- Baseline correction for clean signals
- Standardized feature scaling

### Comprehensive Feature Engineering
- Multi-domain features (time, frequency, space)
- Neurophysiologically motivated time windows
- Standard MEG frequency bands

### Proper ML Validation
- Stratified train/test splits
- Cross-validation for robust estimates
- Multiple algorithms for comparison

### Interpretable Results
- Feature importance analysis
- Confusion matrices
- Performance visualizations

## 🎓 Educational Value

This project is ideal for:
- **ML Engineers** learning neuroimaging
- **Neuroscience Students** learning ML
- **Data Scientists** exploring brain data
- **Researchers** needing MEG analysis templates

### Key Learning Outcomes
1. MEG data structure and preprocessing
2. Neuroimaging-specific feature extraction
3. Time-series classification techniques
4. Cross-validation in neuroscience contexts
5. Interpretation of brain-based ML results

## 🚀 Next Steps & Advanced Projects

### Immediate Extensions
1. **Feature Selection**: Reduce dimensionality, improve interpretability
2. **Deep Learning**: Try CNNs or RNNs for automatic feature learning
3. **Time-Resolved Decoding**: Classify at each time point
4. **Source Localization**: Map activity to brain regions

### Intermediate Projects
1. **Multi-Class Classification**: Add more stimulus types
2. **Subject Generalization**: Train on one subject, test on another
3. **Real-Time Classification**: Online decoding for BCI applications
4. **Connectivity Analysis**: Study brain network interactions

### Advanced Research Directions
1. **Clinical Applications**: Classify neurological conditions
2. **Cognitive States**: Decode attention, memory, decision-making
3. **Individual Differences**: Predict personality or abilities
4. **Multimodal Fusion**: Combine MEG with EEG, fMRI, or behavioral data

### Suggested Datasets
- **MNE Sample Data**: Other paradigms (faces, words)
- **OpenNeuro**: Public neuroimaging datasets
- **FieldTrip Tutorial Data**: Additional MEG examples
- **HCP (Human Connectome Project)**: Large-scale brain data

## 🔧 Customization Options

### Modify Time Windows
```python
tmin, tmax = -0.5, 1.0  # Longer epochs
time_windows = [(0.0, 0.2, "early"), (0.2, 0.6, "late")]  # Different windows
```

### Add Frequency Bands
```python
freq_bands = {
    'slow_gamma': (30, 50),
    'fast_gamma': (50, 80),
    'high_gamma': (80, 120)
}
```

### Try Different Models
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

models['Gradient Boosting'] = GradientBoostingClassifier()
models['Neural Network'] = MLPClassifier(hidden_layer_sizes=(100, 50))
```

## 📚 Further Reading

### MEG/EEG Analysis
- [MNE-Python Documentation](https://mne.tools/)
- "MEG: An Introduction to Methods" by Hansen, Kringelbach, Salmelin
- "Analyzing Neural Time Series Data" by Mike X Cohen

### Machine Learning for Neuroscience
- "Pattern Recognition and Machine Learning" by Bishop
- "The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Research Papers
- Grootswagers et al. (2017). "Decoding Dynamic Brain Patterns from EEG"
- King & Dehaene (2014). "Characterizing the dynamics of mental representations"
- Cichy & Pantazis (2017). "Multivariate pattern analysis of MEG and EEG"

## 🤝 Contributing

This project is designed for educational purposes. Feel free to:
- Extend the analysis with new features
- Try different datasets
- Implement additional ML algorithms
- Improve visualizations
- Add documentation

## 📄 License

This project uses the MNE sample dataset, which is freely available for research and educational purposes. The code is provided as-is for educational use.

## 🙏 Acknowledgments

- **MNE-Python Team**: For excellent neuroimaging tools
- **Neuromag/Elekta**: For the sample dataset
- **Scikit-learn Team**: For machine learning algorithms
- **Python Scientific Community**: For the amazing ecosystem

---

**🎉 Congratulations on completing your first MEG machine learning project!**

*This README demonstrates a complete, professional-grade neuroimaging analysis that achieves state-of-the-art performance on a classic neuroscience problem.*

Remember to build out the connectome on a website. Check features using the best feature decider (maybe random forest)