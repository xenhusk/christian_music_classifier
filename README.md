# Audio-Based Christian Music Classifier

A machine learning project to classify music as Christian or Secular based on **audio features** extracted from music files. Works completely offline without requiring lyrics or internet connection.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the improved model
python improved_audio_classifier.py

# 3. Test model performance
python simple_model_test.py

# 4. Try live demo
python demo_fixed_model.py
```

## Features

- **Audio-Based Classification**: Analyzes audio characteristics instead of lyrics
- **Offline Operation**: No internet connection required
- **Comprehensive Features**: 50+ audio features including tempo, harmony, rhythm, and spectral properties
- **Multiple Models**: Random Forest and SVM classifiers
- **High Coverage**: Can classify ALL audio files in your dataset
- **Visualization**: Confusion matrix and feature importance plots

## Project Structure

```
christian_music_classifier/
├── improved_audio_classifier.py  # 🎯 Main training script (FIXED)
├── demo_fixed_model.py           # 🎵 Live demo of fixed model
├── simple_model_test.py          # 🧪 Model comparison & validation
├── TrainingData/                 # Training data directory
│   ├── ChristianMusic/           # Christian music files (371 files)
│   └── SecularMusic/             # Secular music files (160 files)
├── models/                       # Trained model storage
│   └── improved_audio_classifier_random_forest.joblib  # ✅ Fixed model
├── visualizations/               # Generated plots and charts
├── venv/                         # Python virtual environment
├── requirements.txt              # Python dependencies (CPU-only)
├── MODEL_IMPROVEMENTS.md         # 📋 Detailed fix documentation
└── README.md                     # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Some audio processing libraries may require additional system dependencies:
- **librosa**: Requires FFmpeg
- **essentia**: May require compilation or conda installation

### 2. Prepare Your Audio Files

Organize your music files in the following structure:
```
TrainingData/
├── ChristianMusic/
│   ├── song1.mp3
│   ├── song2.wav
│   └── ...
└── SecularMusic/
    ├── song1.mp3
    ├── song2.wav
    └── ...
```

Supported formats: MP3, WAV, M4A, OPUS, FLAC

## Usage

### 1. Train the Improved Model

```bash
python improved_audio_classifier.py
```

This will:
- Load all audio files from your TrainingData directory
- Extract 65 enhanced audio features from each file
- Apply variance filtering and feature selection (reduces to 30 features)
- Train class-balanced Random Forest and SVM classifiers
- Use cross-validation to select the best model
- Save the improved model to `models/`
- Generate performance visualizations in `visualizations/`


### 2. Test the Improved Model

```bash
python simple_model_test.py
```

This will:
- Compare original vs improved model architectures
- Show performance improvements and bias fixes
- Display detailed model comparison results

### 2.2. Live Demo

```bash
python demo_fixed_model.py
```

This will:
- Load the improved model
- Test random samples from your dataset
- Show real-time predictions with confidence scores
- Demonstrate balanced Christian/Secular classification

## Audio Features

The classifier analyzes these audio characteristics:

### 1. **Spectral Features**
- Spectral centroid (brightness)
- Spectral rolloff (frequency cutoff)
- Spectral bandwidth (frequency spread)
- Spectral contrast (harmonic vs noise)

### 2. **Rhythm Features**
- Tempo (BPM)
- Beat count and patterns
- Zero crossing rate

### 3. **Harmonic Features**
- MFCC coefficients (13 features)
- Chroma features (pitch class)
- Tonnetz features (harmonic content)
- Key estimation and confidence

### 4. **Dynamic Features**
- RMS energy
- Dynamic range
- Harmonic vs percussive ratio

### 5. **Quality Features**
- Spectral flatness
- Sample rate and duration
- Audio file properties

## Model Performance

### Improved Model (v2.0)
- **Coverage**: 100% of audio files (no failed extractions)
- **Test Accuracy**: 84.1% (balanced across classes)
- **Cross-Validation**: 81.8% ± 4.0% (reliable estimate)
- **Christian Detection**: 86% precision, 92% recall
- **Secular Detection**: 78% precision, 66% recall
- **Features**: 30 carefully selected from 65 extracted
- **Class Balancing**: ✅ Handles 2.32:1 data imbalance
- **Speed**: ~4.9 files/second with parallel processing
- **Offline**: No internet connection required

### Key Improvements Over Original
- ✅ Fixed label encoding bias issues
- ✅ Removed constant/low-variance features (11 removed)
- ✅ Added automatic class balancing for imbalanced data
- ✅ Advanced feature selection (65 → 30 features)
- ✅ Cross-validation for reliable performance estimates
- ✅ Enhanced audio feature engineering
- ✅ Proper bias mitigation techniques

## Advantages Over Lyrics-Based Classification

1. **Complete Coverage**: Works with ALL audio files, not just those with lyrics
2. **Offline Operation**: No internet or API dependencies
3. **Language Independent**: Works with any language or instrumental music
4. **Robust**: Doesn't depend on lyrics quality or availability
5. **Fast**: No need to fetch lyrics from external sources

## Troubleshooting

### Common Issues

1. **Audio Loading Errors**
   - Ensure audio files are not corrupted
   - Check file format compatibility
   - Install FFmpeg for MP3 support

2. **Feature Extraction Failures**
   - Check if librosa can load the audio file
   - Verify file permissions
   - Try converting to WAV format

3. **Model Training Issues**
   - Ensure you have both Christian and Secular files
   - Check for sufficient training data (minimum 50 files per class)
   - Verify audio file quality

### Performance Tips

1. **Use WAV files** for fastest processing
2. **Limit duration** to 30 seconds for faster feature extraction
3. **Ensure balanced dataset** (similar number of Christian and Secular files)
4. **Use high-quality audio** for better feature extraction

## Example Output

```
🎵 Audio-Based Christian Music Classifier
==================================================
📁 Loading audio files...
✅ Found 531 audio files
   Christian: 371
   Secular: 160

🔍 Extracting audio features...
✅ Extracted features from 531 files
   Feature dimensions: 50

📊 Data split:
   Training: 424 samples
   Testing: 107 samples

🤖 Training random_forest...
   Training accuracy: 0.892
   Test accuracy: 0.850

🤖 Training svm...
   Training accuracy: 0.875
   Test accuracy: 0.832

🏆 Best model: random_forest (accuracy: 0.850)

📈 Detailed Results:
              precision    recall  f1-score   support
    Christian       0.85      0.88      0.86        64
     Secular       0.85      0.81      0.83        43
    accuracy                           0.85       107
   macro avg       0.85      0.85      0.85       107
weighted avg       0.85      0.85      0.85       107

💾 Model saved to: models/audio_classifier_random_forest.joblib
🎉 Training complete!
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.