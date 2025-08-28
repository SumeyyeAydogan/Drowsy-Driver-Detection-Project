# test_saved_model.py - Test saved model and analyze dataset
import os
import sys
import tensorflow as tf
import numpy as np

# Add src to path
sys.path.append('src')

from src.dataloader import get_data_pipelines
from src.utils import plot_dataset_distribution, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix

def test_saved_model(model_path, data_dir="data", run_dir=None):
    """
    Test a saved model on the dataset
    
    Args:
        model_path: Path to saved model (.h5 file)
        data_dir: Directory containing train/val/test data
        run_dir: Directory to save results (optional)
    """
    print("🚀 Testing Saved Model")
    print("=" * 50)
    
    # 1) Load the saved model
    print(f"📥 Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 2) Load datasets
    print("\n🔄 Loading datasets...")
    try:
        train_ds, val_ds, test_ds = get_data_pipelines(data_dir)
        print("✅ Datasets loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # 3) Plot dataset distribution
    print("\n📊 Analyzing dataset distribution...")
    if run_dir:
        dist_plot_path = os.path.join(run_dir, "plots", "dataset_distribution.png")
        os.makedirs(os.path.dirname(dist_plot_path), exist_ok=True)
        plot_dataset_distribution(data_dir, save_path=dist_plot_path)
    else:
        plot_dataset_distribution(data_dir)
    
    # 4) Evaluate on test set
    print("\n🧪 Evaluating model on test set...")
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for x_batch, y_batch in test_ds:
        # Get predictions
        preds = model.predict(x_batch, verbose=0)
        
        # For binary classification: y_batch is already 0 or 1
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())  # Threshold 0.5
        y_pred_proba.extend(preds.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # 5) Print classification report
    print("\nClassification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=['Not Drowsy', 'Drowsy']))
    
    # 6) Plot evaluation results
    print("\n📈 Plotting evaluation results...")
    
    if run_dir:
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        cm_save_path = os.path.join(plots_dir, "test_confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, save_path=cm_save_path)
        
        # ROC curve
        roc_save_path = os.path.join(plots_dir, "test_roc_curve.png")
        plot_roc_curve(y_true, y_pred_proba, save_path=roc_save_path)
        
        # Precision-Recall curve
        pr_save_path = os.path.join(plots_dir, "test_precision_recall_curve.png")
        plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_save_path)
        
        print(f"✅ All plots saved to: {plots_dir}")
    else:
        # Just display plots without saving
        plot_confusion_matrix(y_true, y_pred)
        plot_roc_curve(y_true, y_pred_proba)
        plot_precision_recall_curve(y_true, y_pred_proba)
    
    # 7) Calculate and print metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    print("\n📊 Model Performance Metrics:")
    print("=" * 40)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_score:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_score': auc_score,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def main():
    """Main function to test saved model"""
    
    # Check if model path is provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default: look for model in runs directory
        runs_dir = "runs"
        if os.path.exists(runs_dir):
            # Find the most recent run
            run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
            if run_dirs:
                latest_run = sorted(run_dirs)[-1]
                model_path = os.path.join(runs_dir, latest_run, "models", "final_model.h5")
                print(f"🔍 Using latest model from: {model_path}")
            else:
                print("❌ No run directories found. Please specify model path.")
                print("Usage: python test_saved_model.py <model_path>")
                return
        else:
            print("❌ No runs directory found. Please specify model path.")
            print("Usage: python test_saved_model.py <model_path>")
            return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create output directory for this test
    test_run_dir = "test_results"
    os.makedirs(test_run_dir, exist_ok=True)
    
    # Test the model
    results = test_saved_model(model_path, run_dir=test_run_dir)
    
    if results:
        print(f"\n🎉 Model testing completed!")
        print(f"📁 Results saved to: {test_run_dir}")
        
        # Save results to JSON
        import json
        results_file = os.path.join(test_run_dir, "test_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'auc_score': float(results['auc_score']),
            'model_path': model_path,
            'test_timestamp': str(np.datetime64('now'))
        }
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
