import os
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
import mlflow
from mlflow.tracking import MlflowClient

# Constants
MONITOR_DATA_DIR = "monitoring_data"
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
CLASS_NAMES = ["setosa", "versicolor", "virginica"]
DRIFT_DETECTION_WINDOW = 100  # Number of predictions to consider for drift detection
PERFORMANCE_WINDOW = 50  # Number of feedback samples to calculate performance metrics

# Initialize monitoring directory
os.makedirs(MONITOR_DATA_DIR, exist_ok=True)
predictions_file = os.path.join(MONITOR_DATA_DIR, "predictions.csv")
feedback_file = os.path.join(MONITOR_DATA_DIR, "feedback.csv")
metrics_file = os.path.join(MONITOR_DATA_DIR, "metrics.json")

# Initialize files if they don't exist
if not os.path.exists(predictions_file):
    pd.DataFrame(columns=["timestamp", "prediction_id"] + FEATURE_NAMES + ["predicted_class", "predicted_class_name", "model_version"]).to_csv(predictions_file, index=False)

if not os.path.exists(feedback_file):
    pd.DataFrame(columns=["timestamp", "prediction_id", "predicted_class", "actual_class", "is_correct"]).to_csv(feedback_file, index=False)

if not os.path.exists(metrics_file):
    with open(metrics_file, 'w') as f:
        json.dump({
            "total_predictions": 0,
            "predictions_with_feedback": 0,
            "accuracy": None,
            "drift_detected": False,
            "last_updated": datetime.datetime.now().isoformat()
        }, f)


def log_prediction(features, prediction, model_version="unknown"):
    """
    Log a prediction to the monitoring system
    
    Args:
        features (list): The input features for the prediction
        prediction (int): The predicted class
        model_version (str): The version of the model that made the prediction
    
    Returns:
        str: A unique ID for this prediction
    """
    # Generate a unique ID for this prediction
    prediction_id = f"pred_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    # Create a record of this prediction
    prediction_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction_id": prediction_id,
        "predicted_class": int(prediction),
        "predicted_class_name": CLASS_NAMES[int(prediction)],
        "model_version": model_version
    }
    
    # Add features
    for i, feature_name in enumerate(FEATURE_NAMES):
        prediction_data[feature_name] = features[i]
    
    # Append to the predictions file
    predictions_df = pd.read_csv(predictions_file)
    predictions_df = pd.concat([predictions_df, pd.DataFrame([prediction_data])], ignore_index=True)
    predictions_df.to_csv(predictions_file, index=False)
    
    # Update metrics
    update_metrics()
    
    return prediction_id


def log_feedback(prediction_id, actual_class):
    """
    Log feedback for a prediction (ground truth)
    
    Args:
        prediction_id (str): The ID of the prediction
        actual_class (int): The actual class
    """
    # Find the prediction
    predictions_df = pd.read_csv(predictions_file)
    prediction_row = predictions_df[predictions_df["prediction_id"] == prediction_id]
    
    if len(prediction_row) == 0:
        print(f"Warning: Prediction ID {prediction_id} not found")
        return
    
    predicted_class = prediction_row["predicted_class"].values[0]
    is_correct = int(predicted_class) == int(actual_class)
    
    # Create a feedback record
    feedback_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prediction_id": prediction_id,
        "predicted_class": predicted_class,
        "actual_class": actual_class,
        "is_correct": is_correct
    }
    
    # Append to the feedback file
    feedback_df = pd.read_csv(feedback_file)
    feedback_df = pd.concat([feedback_df, pd.DataFrame([feedback_data])], ignore_index=True)
    feedback_df.to_csv(feedback_file, index=False)
    
    # Update metrics
    update_metrics()


def detect_drift():
    """
    Detect drift in the predictions
    
    Returns:
        bool: True if drift is detected, False otherwise
    """
    predictions_df = pd.read_csv(predictions_file)
    
    # If we don't have enough predictions yet, return False
    if len(predictions_df) < DRIFT_DETECTION_WINDOW:
        return False
    
    # Get the most recent predictions
    recent_predictions = predictions_df.tail(DRIFT_DETECTION_WINDOW)
    
    # Extract features
    features = recent_predictions[FEATURE_NAMES].values
    
    # Use Isolation Forest to detect anomalies
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies = isolation_forest.fit_predict(features)
    
    # Count anomalies
    anomaly_count = np.sum(anomalies == -1)
    anomaly_rate = anomaly_count / len(anomalies)
    
    # If more than 20% of recent predictions are anomalies, we consider it drift
    return anomaly_rate > 0.2


def calculate_performance_metrics():
    """
    Calculate performance metrics based on feedback
    
    Returns:
        dict: Dictionary of performance metrics
    """
    feedback_df = pd.read_csv(feedback_file)
    
    # If we don't have any feedback yet, return None
    if len(feedback_df) == 0:
        return None
    
    # Get the most recent feedback entries
    recent_feedback = feedback_df.tail(min(len(feedback_df), PERFORMANCE_WINDOW))
    
    # Calculate metrics
    y_true = recent_feedback["actual_class"].values
    y_pred = recent_feedback["predicted_class"].values
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro'),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
    }
    
    # Calculate per-class metrics if we have enough data
    class_metrics = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_samples = (y_true == class_idx).sum()
        if class_samples > 0:
            class_metrics[class_name] = {
                "precision": precision_score(y_true, y_pred, labels=[class_idx], average='micro'),
                "recall": recall_score(y_true, y_pred, labels=[class_idx], average='micro'),
                "f1": f1_score(y_true, y_pred, labels=[class_idx], average='micro'),
                "sample_count": int(class_samples)
            }
    
    metrics["per_class"] = class_metrics
    return metrics


def update_metrics():
    """Update the metrics file with the latest metrics"""
    predictions_df = pd.read_csv(predictions_file)
    feedback_df = pd.read_csv(feedback_file)
    
    # Calculate basic stats
    total_predictions = int(len(predictions_df))
    predictions_with_feedback = int(len(feedback_df))
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics()
    
    # Check for drift
    drift_detected = bool(detect_drift())
    
    # Create metrics object
    metrics = {
        "total_predictions": total_predictions,
        "predictions_with_feedback": predictions_with_feedback,
        "accuracy": float(performance_metrics["accuracy"]) if performance_metrics and performance_metrics["accuracy"] is not None else None,
        "precision": float(performance_metrics["precision_macro"]) if performance_metrics and performance_metrics["precision_macro"] is not None else None,
        "recall": float(performance_metrics["recall_macro"]) if performance_metrics and performance_metrics["recall_macro"] is not None else None,
        "f1": float(performance_metrics["f1_macro"]) if performance_metrics and performance_metrics["f1_macro"] is not None else None,
        "drift_detected": drift_detected,
        "last_updated": datetime.datetime.now().isoformat()
    }
    
    # Add per-class metrics if available
    if performance_metrics and "per_class" in performance_metrics:
        per_class_metrics = {}
        for class_name, class_metrics in performance_metrics["per_class"].items():
            per_class_metrics[class_name] = {
                "precision": float(class_metrics["precision"]) if class_metrics["precision"] is not None else None,
                "recall": float(class_metrics["recall"]) if class_metrics["recall"] is not None else None,
                "f1": float(class_metrics["f1"]) if class_metrics["f1"] is not None else None,
                "sample_count": int(class_metrics["sample_count"])
            }
        metrics["per_class"] = per_class_metrics
    
    # Save metrics to file
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def get_current_metrics():
    """
    Get the current monitoring metrics
    
    Returns:
        dict: Dictionary of monitoring metrics
    """
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return {}


def visualize_monitoring_results(output_dir="monitoring_visualizations"):
    """
    Generate visualizations of monitoring results
    
    Args:
        output_dir (str): Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    predictions_df = pd.read_csv(predictions_file)
    feedback_df = pd.read_csv(feedback_file)
    
    if len(predictions_df) == 0:
        print("No prediction data available for visualization")
        return
    
    # 1. Prediction Distribution
    plt.figure(figsize=(10, 6))
    class_counts = predictions_df["predicted_class_name"].value_counts()
    plt.bar(class_counts.index, class_counts.values)
    plt.title("Distribution of Predictions")
    plt.ylabel("Count")
    plt.xlabel("Class")
    plt.savefig(os.path.join(output_dir, "prediction_distribution.png"))
    plt.close()
    
    # 2. Feature Distributions
    plt.figure(figsize=(16, 10))
    for i, feature in enumerate(FEATURE_NAMES):
        plt.subplot(2, 2, i+1)
        for class_name in CLASS_NAMES:
            class_data = predictions_df[predictions_df["predicted_class_name"] == class_name][feature]
            if len(class_data) > 0:
                plt.hist(class_data, alpha=0.5, label=class_name, bins=15)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    plt.close()
    
    # 3. Accuracy over time (if feedback available)
    if len(feedback_df) > 10:
        plt.figure(figsize=(12, 6))
        
        # Convert timestamps to datetime
        feedback_df["timestamp"] = pd.to_datetime(feedback_df["timestamp"])
        
        # Sort by timestamp
        feedback_df = feedback_df.sort_values("timestamp")
        
        # Calculate rolling accuracy
        feedback_df["is_correct"] = feedback_df["is_correct"].astype(int)
        rolling_acc = feedback_df["is_correct"].rolling(window=min(10, len(feedback_df)), min_periods=1).mean()
        
        plt.plot(feedback_df["timestamp"], rolling_acc)
        plt.title("Model Accuracy Over Time (10-sample rolling window)")
        plt.ylabel("Accuracy")
        plt.xlabel("Time")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_over_time.png"))
        plt.close()
    
    # 4. Confusion matrix (if feedback available)
    if len(feedback_df) > 0:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Get actual and predicted classes
        y_true = feedback_df["actual_class"].values
        y_pred = feedback_df["predicted_class"].values
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Confusion Matrix")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def check_retraining_needed():
    """
    Check if retraining is needed based on monitoring metrics
    
    Returns:
        bool: True if retraining is recommended, False otherwise
    """
    metrics = get_current_metrics()
    
    # Criteria for retraining:
    # 1. Drift detected
    # 2. Sufficient feedback data (at least 50 samples)
    # 3. Performance below threshold (accuracy < 0.9)
    
    if metrics.get("drift_detected", False):
        return True
    
    if metrics.get("predictions_with_feedback", 0) >= 50:
        if metrics.get("accuracy", 1.0) < 0.9:
            return True
    
    return False


def trigger_retraining():
    """
    Trigger model retraining based on monitoring results
    """
    print("Triggering model retraining...")
    # In a real-world scenario, this would trigger a CI/CD pipeline
    # For this demo, we'll just run the training pipeline directly
    from simple_ml_pipeline import run_pipeline
    run_pipeline()
    
    # Register the new model
    from model_registry import register_model
    model_details = register_model()
    
    # Log the retraining event
    with open(os.path.join(MONITOR_DATA_DIR, "retraining_events.txt"), "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} - Retrained model registered as version {model_details.version}\n")
    
    return model_details


def generate_monitoring_report():
    """
    Generate a monitoring report with key metrics and visualizations
    
    Returns:
        str: HTML report
    """
    metrics = get_current_metrics()
    
    # Generate visualizations
    visualize_monitoring_results()
    
    # Format accuracy and other metrics safely
    accuracy_display = f"{metrics.get('accuracy'):.4f}" if metrics.get('accuracy') is not None else "N/A"
    precision_display = f"{metrics.get('precision'):.4f}" if metrics.get('precision') is not None else "N/A"
    recall_display = f"{metrics.get('recall'):.4f}" if metrics.get('recall') is not None else "N/A"
    f1_display = f"{metrics.get('f1'):.4f}" if metrics.get('f1') is not None else "N/A"
    
    # Create HTML report
    report = f"""
    <html>
    <head>
        <title>Model Monitoring Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .metric-card {{ background: #f9f9f9; padding: 20px; margin-bottom: 20px; border-radius: 10px; }}
            .warning {{ background: #fff3cd; }}
            h1, h2 {{ color: #333; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
            .metric {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
            .metric h3 {{ margin-top: 0; }}
            .metric p {{ font-size: 24px; font-weight: bold; }}
            img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Monitoring Report</h1>
            <p>Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="metric-card {'warning' if metrics.get('drift_detected', False) else ''}">
                <h2>Overview</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <h3>Total Predictions</h3>
                        <p>{metrics.get('total_predictions', 0)}</p>
                    </div>
                    <div class="metric">
                        <h3>Predictions with Feedback</h3>
                        <p>{metrics.get('predictions_with_feedback', 0)}</p>
                    </div>
                    <div class="metric">
                        <h3>Accuracy</h3>
                        <p>{accuracy_display}</p>
                    </div>
                    <div class="metric">
                        <h3>Drift Detected</h3>
                        <p style="color: {'red' if metrics.get('drift_detected', False) else 'green'}">
                            {metrics.get('drift_detected', False)}
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="metric-card">
                <h2>Performance Metrics</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <h3>Precision (macro)</h3>
                        <p>{precision_display}</p>
                    </div>
                    <div class="metric">
                        <h3>Recall (macro)</h3>
                        <p>{recall_display}</p>
                    </div>
                    <div class="metric">
                        <h3>F1 Score (macro)</h3>
                        <p>{f1_display}</p>
                    </div>
                </div>
                
                <h3>Class-level Performance</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Class</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Precision</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Recall</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">F1</th>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Samples</th>
                    </tr>
    """
    
    # Add per-class metrics if available
    if "per_class" in metrics:
        for class_name, class_metrics in metrics["per_class"].items():
            # Format class metrics safely
            class_precision = f"{class_metrics.get('precision'):.4f}" if class_metrics.get('precision') is not None else "N/A"
            class_recall = f"{class_metrics.get('recall'):.4f}" if class_metrics.get('recall') is not None else "N/A"
            class_f1 = f"{class_metrics.get('f1'):.4f}" if class_metrics.get('f1') is not None else "N/A"
            class_samples = class_metrics.get('sample_count', 0)
            
            report += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_name}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_precision}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_recall}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_f1}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{class_samples}</td>
                    </tr>
            """
    
    report += """
                </table>
            </div>
            
            <div class="metric-card">
                <h2>Visualizations</h2>
                
                <h3>Prediction Distribution</h3>
                <img src="monitoring_visualizations/prediction_distribution.png" alt="Prediction Distribution">
                
                <h3>Feature Distributions</h3>
                <img src="monitoring_visualizations/feature_distributions.png" alt="Feature Distributions">
    """
    
    # Add accuracy over time chart if available
    if os.path.exists("monitoring_visualizations/accuracy_over_time.png"):
        report += """
                <h3>Accuracy Over Time</h3>
                <img src="monitoring_visualizations/accuracy_over_time.png" alt="Accuracy Over Time">
        """
    
    # Add confusion matrix if available
    if os.path.exists("monitoring_visualizations/confusion_matrix.png"):
        report += """
                <h3>Confusion Matrix</h3>
                <img src="monitoring_visualizations/confusion_matrix.png" alt="Confusion Matrix">
        """
    
    report += """
            </div>
            
            <div class="metric-card">
                <h2>Retraining Recommendation</h2>
    """
    
    if check_retraining_needed():
        report += """
                <p style="color: red; font-weight: bold;">Model retraining is recommended based on current metrics.</p>
                <ul>
        """
        
        if metrics.get("drift_detected", False):
            report += """
                    <li>Drift has been detected in the incoming data.</li>
            """
        
        if metrics.get("accuracy", 1.0) < 0.9 and metrics.get("predictions_with_feedback", 0) >= 50:
            accuracy_value = f"{metrics.get('accuracy', 0):.4f}"
            report += f"""
                    <li>Model accuracy ({accuracy_value}) is below the threshold of 0.9.</li>
            """
        
        report += """
                </ul>
        """
    else:
        report += """
                <p style="color: green; font-weight: bold;">Model performance is acceptable. No retraining needed at this time.</p>
        """
    
    report += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save report to file
    with open("model_monitoring_report.html", "w") as f:
        f.write(report)
    
    return report


if __name__ == "__main__":
    # Demo code to simulate using the monitoring system
    # In a real application, these functions would be called from your model serving API
    
    # 1. Generate some random predictions and log them
    from sklearn.datasets import load_iris
    
    print("Simulating predictions and feedback...")
    
    # Load Iris dataset for demo
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Simulate 50 predictions with user feedback
    prediction_ids = []
    for i in range(50):
        # Get a random sample from Iris
        idx = np.random.randint(0, len(X))
        features = X[idx]
        true_class = y[idx]
        
        # Simulate a model prediction (sometimes correct, sometimes wrong)
        if np.random.random() > 0.2:  # 80% accuracy
            predicted_class = true_class
        else:
            # Make a wrong prediction
            wrong_classes = [c for c in range(3) if c != true_class]
            predicted_class = np.random.choice(wrong_classes)
        
        # Log the prediction
        prediction_id = log_prediction(features, predicted_class, model_version="v1")
        prediction_ids.append((prediction_id, true_class))
    
    # Log feedback for some of the predictions
    for i, (prediction_id, true_class) in enumerate(prediction_ids):
        # Only log feedback for some predictions to simulate real-world scenario
        if np.random.random() > 0.3:  # 70% of predictions get feedback
            log_feedback(prediction_id, true_class)
    
    # Generate monitoring visualizations and report
    print("Generating monitoring visualizations...")
    visualize_monitoring_results()
    
    print("Generating monitoring report...")
    generate_monitoring_report()
    
    # Check if retraining is needed
    if check_retraining_needed():
        print("Retraining is recommended.")
    else:
        print("No retraining needed at this time.")
    
    print(f"Monitoring report saved to model_monitoring_report.html")
    print(f"Monitoring data saved to {MONITOR_DATA_DIR}/") 