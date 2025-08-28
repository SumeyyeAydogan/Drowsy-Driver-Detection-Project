import os

#from src.split_data      import split_dataset
from src.dataloader      import get_data_pipelines
from src.model           import build_model
from src.train           import train_model
from src.utils           import plot_history, plot_metrics, create_run_directories
from src.evaluate        import evaluate_model
from src.export          import save_model

if __name__ == "__main__":
    print("🚀 Starting Drowsy Driver Detection Project...")
    print("=" * 50)
    
    # Project root directory: the folder where this file is located
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    #project_root = r"D:\internship\Drowsy-Driver-Detection-Project"

    # 1) Raw data folder (what you have)
    #raw_dir = os.path.join(project_root, "train_data")
    #if not os.path.isdir(raw_dir):
    #    raise FileNotFoundError(f"`dataset_raw` not found: {raw_dir}")

    # 2) Folder where split data will go
    output_dir = os.path.join(project_root, "data")
    print(f"📁 Data directory: {output_dir}")

    # 3) Create Train/Val/Test folder hierarchy
    #    raw_dir contains => drowsy, notdrowsy
    #split_dataset(raw_dir, output_dir)

    # 4) Create run directories
    print("📁 Creating run directories...")
    run_dir = create_run_directories("run_001")
    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    print(f"✅ Run directories created: {run_dir}")

    # 5) tf.data pipelines
    print("🔄 Loading datasets...")
    train_ds, val_ds, test_ds = get_data_pipelines(output_dir)
    print("✅ Datasets loaded successfully!")

    # 6) Build and train model
    print("🏗️  Building model...")
    model = build_model()
    print("✅ Model built successfully!")
    
    print("🎯 Starting training...")
    history = train_model(model, train_ds, val_ds, epochs=2)
    print("✅ Training completed!")

    # 7) Plot training graphs and save them
    print("📊 Plotting training history...")
    history_plot_path = os.path.join(plots_dir, "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("📈 Plotting metrics...")
    metrics_plot_path = os.path.join(plots_dir, "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    # 8) Evaluate on test set
    print("🧪 Evaluating model on test set...")
    evaluate_model(model, test_ds, plots_dir=plots_dir)
    print("✅ Model evaluation completed!")

    # 9) Save model to run directory
    print("💾 Saving model...")
    model_path = os.path.join(models_dir, "final_model.h5")
    save_model(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # 10) Save simple config
    config = {
        "run_name": "run_001",
        "epochs": 2,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": ["notdrowsy", "drowsy"]
    }
    
    import json
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to: {config_path}")
    
    print("\n" + "=" * 50)
    print("🎉 All tasks completed successfully!")
    print(f"📁 Results saved to: {run_dir}")
    print("Project finished! 🚀")