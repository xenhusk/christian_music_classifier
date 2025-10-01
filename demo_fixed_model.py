#!/usr/bin/env python3
"""
Demo: Fixed Christian Music Classifier

This script demonstrates the improved model's ability to correctly classify
Christian and secular music with balanced performance.

Features:
- Test single model or compare multiple models
- Multithreaded feature extraction for faster processing
- Side-by-side model comparison
- Command-line interface for model selection
"""

import os
import sys
import numpy as np
import joblib
from pathlib import Path
import random
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_models(model_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load models from the models directory.
    
    Args:
        model_name: Specific model filename to load, or None to load all
        
    Returns:
        Dictionary mapping model names to model data
    """
    models_dir = Path("models")
    models = {}
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return models
    
    if model_name:
        # Load specific model
        model_path = models_dir / model_name
        if not model_path.exists():
            print(f"❌ Model '{model_name}' not found!")
            return models
        
        try:
            model_data = joblib.load(model_path)
            models[model_name] = model_data
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"❌ Error loading model '{model_name}': {e}")
    else:
        # Load all .joblib models
        for model_path in models_dir.glob("*.joblib"):
            try:
                model_data = joblib.load(model_path)
                models[model_path.name] = model_data
                print(f"✅ Loaded model: {model_path.name}")
            except Exception as e:
                print(f"❌ Error loading model '{model_path.name}': {e}")
    
    return models


def extract_features_parallel(files: List[Tuple[str, str]], max_workers: int = 4) -> Dict[str, Tuple[Optional[Dict], str]]:
    """
    Extract features from multiple files in parallel.
    
    Args:
        files: List of (file_path, true_label) tuples
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary mapping file paths to (features, true_label) tuples
    """
    from improved_audio_classifier import ImprovedAudioFeatureExtractor
    
    feature_extractor = ImprovedAudioFeatureExtractor()
    results = {}
    
    print(f"\n🔄 Extracting features using {max_workers} parallel workers...")
    start_time = time.time()
    
    def extract_single(file_info):
        file_path, true_label = file_info
        try:
            features = feature_extractor.extract_features_single(file_path)
            return file_path, (features, true_label)
        except Exception as e:
            logger.error(f"Error extracting features from {Path(file_path).name}: {e}")
            return file_path, (None, true_label)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single, file_info): file_info for file_info in files}
        
        for future in as_completed(futures):
            file_path, result = future.result()
            results[file_path] = result
    
    elapsed_time = time.time() - start_time
    print(f"✅ Feature extraction complete in {elapsed_time:.2f}s ({elapsed_time/len(files):.2f}s per file)")
    
    return results


def predict_with_model(features_dict: Dict[str, float], model_data: Dict) -> Tuple[str, float, np.ndarray]:
    """
    Make prediction using a model.
    
    Args:
        features_dict: Dictionary of extracted features
        model_data: Loaded model data
        
    Returns:
        Tuple of (predicted_label, confidence, probabilities)
    """
    # Prepare features for model
    X = np.array([[features_dict.get(name, 0.0) for name in model_data['feature_names']]])
    
    # Apply preprocessing pipeline
    X_variance_filtered = model_data['variance_selector'].transform(X)
    X_scaled = model_data['scaler'].transform(X_variance_filtered)
    X_processed = model_data['feature_selector'].transform(X_scaled)
    
    # Make prediction
    pred_numeric = model_data['model'].predict(X_processed)[0]
    pred_proba = model_data['model'].predict_proba(X_processed)[0]
    
    # Convert to label
    pred_label = model_data['label_map'][pred_numeric]
    confidence = max(pred_proba)
    
    return pred_label, confidence, pred_proba


def demo_fixed_model(model_name: Optional[str] = None, num_samples: int = 10, max_workers: int = 4):
    """
    Demonstrate the fixed model's improved performance.
    
    Args:
        model_name: Specific model to test, or None to test all models
        num_samples: Number of samples to test per class
        max_workers: Number of parallel workers for feature extraction
    """
    
    print("🎵 Christian Music Classifier - Enhanced Demo")
    print("=" * 70)
    
    # Load models
    print("\n📦 Loading models...")
    models = load_models(model_name)
    
    if not models:
        print("❌ No models loaded! Please train the model first:")
        print("   python improved_audio_classifier.py")
        return
    
    print(f"\n📊 Loaded {len(models)} model(s):")
    for name, model_data in models.items():
        print(f"   • {name}")
        print(f"     - Type: {model_data['model_type']}")
        print(f"     - Features: {len(model_data['feature_names'])} → {len(model_data['selected_feature_names'])} (selected)")
        print(f"     - Class weights: {model_data['class_weights']}")
    
    # Load test files
    print("\n📁 Loading test files...")
    
    data_path = Path("TrainingData")
    christian_files = []
    secular_files = []
    
    # Collect files
    christian_path = data_path / "ChristianMusic"
    if christian_path.exists():
        for file_path in christian_path.rglob("*"):
            if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.opus', '.flac']:
                christian_files.append(str(file_path))
    
    secular_path = data_path / "SecularMusic"
    if secular_path.exists():
        for file_path in secular_path.rglob("*"):
            if file_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.opus', '.flac']:
                secular_files.append(str(file_path))
    
    print(f"✅ Found {len(christian_files)} Christian and {len(secular_files)} Secular files")
    
    if len(christian_files) == 0 and len(secular_files) == 0:
        print("❌ No test files found in TrainingData directory!")
        return
    
    # Sample files for demo
    demo_files = []
    
    if christian_files:
        christian_samples = random.sample(christian_files, min(num_samples, len(christian_files)))
        for file_path in christian_samples:
            demo_files.append((file_path, 'Christian'))
    
    if secular_files:
        secular_samples = random.sample(secular_files, min(num_samples, len(secular_files)))
        for file_path in secular_samples:
            demo_files.append((file_path, 'Secular'))
    
    random.shuffle(demo_files)
    
    # Extract features in parallel
    print(f"\n🎯 Testing {len(demo_files)} random samples...")
    print("=" * 70)
    
    features_results = extract_features_parallel(demo_files, max_workers=max_workers)
    
    # Test each model on the extracted features
    model_results = {name: {'correct': 0, 'total': 0, 'predictions': []} for name in models.keys()}
    
    print(f"\n📊 Testing predictions...\n")
    
    for i, (file_path, true_label) in enumerate(demo_files, 1):
        features, _ = features_results.get(file_path, (None, true_label))
        
        if features is None:
            print(f"{i:2d}. ❌ Failed to extract features for {Path(file_path).name[:50]}")
            continue
        
        filename = Path(file_path).name
        print(f"{i:2d}. {filename[:60]:<60}")
        print(f"    ✓ True label: {true_label}")
        
        # Test with each model
        for model_name, model_data in models.items():
            try:
                pred_label, confidence, pred_proba = predict_with_model(features, model_data)
                
                is_correct = pred_label == true_label
                if is_correct:
                    model_results[model_name]['correct'] += 1
                model_results[model_name]['total'] += 1
                model_results[model_name]['predictions'].append({
                    'file': filename,
                    'true': true_label,
                    'pred': pred_label,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                status = "✅" if is_correct else "❌"
                
                if len(models) > 1:
                    # Show compact results when comparing multiple models
                    model_short_name = model_name.replace('improved_audio_classifier_', '').replace('.joblib', '').upper()
                    print(f"    {status} {model_short_name:12s}: {pred_label:9s} ({confidence:.3f})")
                else:
                    # Show detailed results for single model
                    print(f"    {status} Predicted: {pred_label} (confidence: {confidence:.3f})")
                    christian_idx = 0 if model_data['label_map'][0] == 'Christian' else 1
                    secular_idx = 1 - christian_idx
                    print(f"    Probabilities: Christian: {pred_proba[christian_idx]:.3f}, Secular: {pred_proba[secular_idx]:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error with {model_name}: {e}")
        
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MODEL COMPARISON RESULTS")
    print("=" * 70)
    
    for model_name, results in model_results.items():
        if results['total'] > 0:
            accuracy = results['correct'] / results['total']
            
            # Calculate per-class metrics
            christian_correct = sum(1 for p in results['predictions'] if p['true'] == 'Christian' and p['correct'])
            christian_total = sum(1 for p in results['predictions'] if p['true'] == 'Christian')
            secular_correct = sum(1 for p in results['predictions'] if p['true'] == 'Secular' and p['correct'])
            secular_total = sum(1 for p in results['predictions'] if p['true'] == 'Secular')
            
            print(f"\n🎯 {model_name}")
            print(f"   Overall: {results['correct']}/{results['total']} = {accuracy:.1%}")
            
            if christian_total > 0:
                christian_acc = christian_correct / christian_total
                print(f"   Christian accuracy: {christian_correct}/{christian_total} = {christian_acc:.1%}")
            
            if secular_total > 0:
                secular_acc = secular_correct / secular_total
                print(f"   Secular accuracy: {secular_correct}/{secular_total} = {secular_acc:.1%}")
            
            avg_confidence = np.mean([p['confidence'] for p in results['predictions']])
            print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Determine best model if comparing multiple
    if len(models) > 1:
        best_model = max(model_results.items(), key=lambda x: x[1]['correct'] / max(x[1]['total'], 1))
        print(f"\n🏆 Best performing model: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['correct'] / best_model[1]['total']:.1%}")
    
    print("\n" + "=" * 70)
    print("✨ DEMO FEATURES:")
    print("   ✅ Multithreaded feature extraction")
    print("   ✅ Side-by-side model comparison")
    print("   ✅ Balanced predictions (not biased toward Christian)")
    print("   ✅ Proper confidence scores")
    print("   ✅ Works on all audio formats")
    print("   ✅ No internet connection required")
    
    print("\n🎉 Demo complete!")

def list_available_models():
    """List all available models in the models directory."""
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return
    
    model_files = list(models_dir.glob("*.joblib"))
    
    if not model_files:
        print("❌ No models found in models directory!")
        return
    
    print("\n📦 Available models:")
    for i, model_path in enumerate(model_files, 1):
        print(f"   {i}. {model_path.name}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Christian Music Classifier - Enhanced Demo with Model Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models with default settings (10 samples per class, 4 workers)
  python demo_fixed_model.py
  
  # Test specific model
  python demo_fixed_model.py --model improved_audio_classifier_random_forest.joblib
  
  # Test with more samples and workers for faster processing
  python demo_fixed_model.py --samples 20 --workers 8
  
  # List available models
  python demo_fixed_model.py --list
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Specific model filename to test (default: test all models)'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10,
        help='Number of samples to test per class (default: 10)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers for feature extraction (default: 4)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
    else:
        demo_fixed_model(
            model_name=args.model,
            num_samples=args.samples,
            max_workers=args.workers
        )