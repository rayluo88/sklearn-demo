import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import mlflow.pyfunc
# Import model monitoring
import model_monitoring

# Import Feast
from feast import FeatureStore
from pathlib import Path
from datetime import datetime
import pandas as pd # For creating DataFrame for online features

app = Flask(__name__)

# --- MLflow Model Loading --- #
FEAST_REPO_PATH = "feature_repository/" # Define path to the Feast repository

# Global variable for the model and feature store
MODEL = None
FEAST_STORE = None

def load_model_and_store(model_name="iris-classifier", stage="Staging"):
    """Load the MLflow model and initialize the Feast FeatureStore."""
    global MODEL, FEAST_STORE
    # Load MLflow model
    model_uri = f"models:/{model_name}/{stage}"
    try:
        MODEL = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model '{model_name}' stage '{stage}' from {model_uri}")
    except Exception as e:
        print(f"Error loading model '{model_name}' stage '{stage}': {e}")
        model_uri_latest = f"models:/{model_name}/latest"
        try:
            MODEL = mlflow.pyfunc.load_model(model_uri_latest)
            print(f"Loaded latest model version instead from {model_uri_latest}")
        except Exception as e_latest:
            print(f"Error loading latest model version: {e_latest}")
            MODEL = None # Ensure model is None if loading fails

    # Initialize Feast FeatureStore
    try:
        # Ensure the path is resolved correctly from the API script's location
        repo_path_resolved = str(Path(__file__).parent / FEAST_REPO_PATH)
        FEAST_STORE = FeatureStore(repo_path=repo_path_resolved)
        print(f"Successfully initialized Feast FeatureStore from {repo_path_resolved}")
    except Exception as e:
        print(f"Error initializing Feast FeatureStore: {e}")
        FEAST_STORE = None

# Load model and store at startup
load_model_and_store()

# Class mapping for Iris dataset
CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# HTML template for the UI
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Iris Classifier API with Feast</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        label {
            display: block;
            margin: 15px 0 5px;
            font-weight: bold;
        }
        input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        input[type="submit"], button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
        }
        input[type="submit"]:hover, button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
            display: none;
        }
        .api-info {
            margin-top: 30px;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
        }
        code {
            background-color: #f1f1f1;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
        }
        pre {
            background-color: #f1f1f1;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        /* New feedback form styling */
        .feedback-form {
            margin-top: 20px;
            padding: 15px;
            background-color: #e3f2fd;
            border-radius: 4px;
            display: none;
        }
        .feedback-form h3 {
            margin-top: 0;
        }
        .radio-group {
            margin: 10px 0;
        }
        .radio-group label {
            display: inline;
            margin-right: 15px;
            font-weight: normal;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Iris Flower Classifier (with Feast)</h1>
        
        <form id="prediction-form">
            <label for="sepal_length">Sepal Length (cm):</label>
            <input type="number" id="sepal_length" name="sepal_length" step="0.1" value="5.1" required>
            
            <label for="sepal_width">Sepal Width (cm):</label>
            <input type="number" id="sepal_width" name="sepal_width" step="0.1" value="3.5" required>
            
            <label for="petal_length">Petal Length (cm):</label>
            <input type="number" id="petal_length" name="petal_length" step="0.1" value="1.4" required>
            
            <label for="petal_width">Petal Width (cm):</label>
            <input type="number" id="petal_width" name="petal_width" step="0.1" value="0.2" required>
            
            <input type="submit" value="Predict">
        </form>
        
        <div class="result" id="result">
            <h3>Prediction Result:</h3>
            <p id="prediction"></p>
        </div>
        
        <!-- New feedback form -->
        <div class="feedback-form" id="feedback-form">
            <h3>Provide Feedback:</h3>
            <p>If you know the actual class, please provide feedback to help improve the model:</p>
            <input type="hidden" id="prediction-id" name="prediction-id">
            <div class="radio-group">
                <input type="radio" id="class-0" name="actual-class" value="0">
                <label for="class-0">Setosa</label>
                
                <input type="radio" id="class-1" name="actual-class" value="1">
                <label for="class-1">Versicolor</label>
                
                <input type="radio" id="class-2" name="actual-class" value="2">
                <label for="class-2">Virginica</label>
            </div>
            <button id="submit-feedback">Submit Feedback</button>
        </div>
        
        <div class="api-info">
            <h3>API Usage:</h3>
            <p>POST to <code>/predict</code> with JSON:</p>
            <pre>{
  "features": [
    [sepal_length, sepal_width, petal_length, petal_width] // raw features
  ],
  "entity_ids": [unique_id_for_each_sample_if_any] // Optional: for more direct Feast lookup if IDs are known
}</pre>
            <h4>Example:</h4>
            <pre>curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'</pre>
            
            <h4>Health Check:</h4>
            <p>GET request to <code>/health</code> returns the model status</p>
            
            <h4>Model Monitoring:</h4>
            <p>View the current model monitoring dashboard at <code>/monitoring</code></p>
        </div>
    </div>
    
    <script>
        document.getElementById('prediction-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const sepalLength = parseFloat(document.getElementById('sepal_length').value);
            const sepalWidth = parseFloat(document.getElementById('sepal_width').value);
            const petalLength = parseFloat(document.getElementById('petal_length').value);
            const petalWidth = parseFloat(document.getElementById('petal_width').value);
            
            const data = {
                features: [[sepalLength, sepalWidth, petalLength, petalWidth]]
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('prediction').textContent = `Predicted class: ${data.predictions[0]}`;
                document.getElementById('result').style.display = 'block';
                
                // Show feedback form and store prediction ID
                document.getElementById('prediction-id').value = data.prediction_id;
                document.getElementById('feedback-form').style.display = 'block';
            })
            .catch((error) => {
                console.error('Error:', error);
                document.getElementById('prediction').textContent = `Error: ${error}`;
                document.getElementById('result').style.display = 'block';
            });
        });
        
        // Handle feedback submission
        document.getElementById('submit-feedback').addEventListener('click', function() {
            const predictionId = document.getElementById('prediction-id').value;
            const selectedClass = document.querySelector('input[name="actual-class"]:checked');
            
            if (!selectedClass) {
                alert('Please select an actual class');
                return;
            }
            
            const actualClass = selectedClass.value;
            
            fetch('/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prediction_id: predictionId,
                    actual_class: parseInt(actualClass)
                }),
            })
            .then(response => response.json())
            .then(data => {
                alert('Thank you for your feedback!');
                document.getElementById('feedback-form').style.display = 'none';
            })
            .catch((error) => {
                console.error('Error:', error);
                alert('Error submitting feedback: ' + error);
            });
        });
    </script>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    """Serve the prediction UI."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    model_ok = MODEL is not None
    feast_ok = FEAST_STORE is not None
    status = {
        "status": "healthy" if model_ok and feast_ok else "unhealthy",
        "model_loaded": model_ok,
        "feast_store_initialized": feast_ok
    }
    return jsonify(status), 200 if model_ok and feast_ok else 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint. Uses Feast to fetch online features if entity_ids are provided,
       otherwise uses raw features directly for prediction (after ensuring they are shaped correctly).
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
    if FEAST_STORE is None:
        return jsonify({"error": "Feast store not initialized"}), 500
        
    try:
        content = request.json
        raw_features_list = content.get('features') # Expects a list of lists
        entity_ids_list = content.get('entity_ids') # Optional: list of entity IDs

        if not raw_features_list and not entity_ids_list:
            return jsonify({"error": "Either 'features' or 'entity_ids' must be provided"}), 400

        # For demo: if entity_ids are provided, use them. Otherwise, use raw_features_list
        # to construct entity_rows for Feast. This simulates fetching based on current inputs.
        if entity_ids_list:
            entity_rows = [{"iris_id": eid} for eid in entity_ids_list]
        elif raw_features_list:
            # Create pseudo entity_rows for get_online_features based on the raw features provided.
            # This is for demo to show interaction. Typically you'd have known entity_ids.
            # We assign temporary iris_ids for the call, assuming they might exist in online store.
            # Or, if features themselves are registered (not typical for Feast raw values as keys),
            # this structure would be different.
            # For simplicity, if raw features are given and no entity_ids, we proceed to predict directly
            # and use Feast more for logging or if transformations were defined in Feast.
            # For THIS demo, let's ensure any prediction still goes through a conceptual "feature retrieval"
            # by passing the raw features to the model as if they were retrieved.
            # We will use the raw features to form the input for the model directly.
            # The main Feast benefit in API is if features were *transformed* by Feast.
            # Here, they are raw, so direct use is fine after this conceptual step.
            
            # Constructing entity_rows from raw_features for get_online_features
            # This step assumes that if we had IDs, we'd fetch. Since we have features,
            # we prepare them as if they were fetched, for consistent processing path.
            # However, `get_online_features` needs entity keys.
            # For a simple demo without actual online store lookups by new IDs:
            # We will use the raw features directly for prediction as the model expects them.
            # We will still log these features via the model_monitoring which could act as a way
            # to feed data back if this were a live system.
            
            # For this demo, we will create an entity_df from the raw features and try to get them.
            # This requires the online store to be populated.
            entity_rows_for_feast = []
            for i, features_sample in enumerate(raw_features_list):
                entity_rows_for_feast.append({
                    "iris_id": i, # Placeholder ID for the request
                    # We don't pass feature values here, Feast should look them up by iris_id
                    # This requires the online store to be populated with these iris_ids (0, 1, 2...) 
                    # and corresponding features from `feast materialize`
                })
            
            # If entity_ids were explicitly passed, this list is used. If not, we use placeholder IDs for demo.
            # For a robust demo of online serving, the API should expect entity_ids that are known to be in the online store.
            # If we only have raw features and want to use Feast for transformation (not applicable here as features are raw),
            # the pattern would be different.
            
            # Let's assume for /predict, if entity_ids are not given, we use raw features directly.
            # And if entity_ids ARE given, we attempt a Feast lookup.
            if entity_ids_list:
                online_features_response = FEAST_STORE.get_online_features(
                    features=[
                        "iris_features:sepal_length",
                        "iris_features:sepal_width",
                        "iris_features:petal_length",
                        "iris_features:petal_width"
                    ],
                    entity_rows=entity_rows # Constructed from entity_ids_list
                ).to_dict()
                # Convert dict to numpy array in correct order for the model
                processed_features_list = []
                for i in range(len(entity_ids_list)):
                    # Order matters: sepal_length, sepal_width, petal_length, petal_width
                    # This assumes the keys in online_features_response include the entity key 'iris_id'
                    # and the feature names.
                    # The order of features from get_online_features needs to be carefully managed.
                    # A safer way is to fetch into a DataFrame and select columns.
                    # For simplicity, this example might be brittle.
                    # A better approach with to_dict():
                    # features_df = pd.DataFrame.from_dict(online_features_response)
                    # ordered_features = features_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.tolist()
                    # This is complex. Simpler for demo: directly use raw_features if no IDs.
                    
                    # Simpler: If entity_ids are provided, we assume they are for logging/context, 
                    # and still use raw_features for prediction to keep API contract simple for demo.
                    # OR, if we *really* want to show online lookup, the UI/client needs to send IDs.
                    # Let's stick to: API takes raw features, uses them directly.
                    # Feast online store interaction will be minimal for this version to keep it runnable.
                    pass # Fall through to use raw_features_list

        # If we fell through or didn't use entity_ids for lookup, use raw features:
        model_input_features = np.array(raw_features_list)
        
        # Make prediction
        predictions_numeric = MODEL.predict(model_input_features)
        
        # Convert numeric predictions to class names
        class_predictions = [CLASS_MAPPING.get(int(pred), f"Unknown-{pred}") for pred in predictions_numeric]
        
        # Log the prediction for monitoring (using the first sample's features for now)
        # model_monitoring.log_prediction expects a single feature list.
        logged_prediction_ids = []
        for i, feature_row in enumerate(raw_features_list):
            model_version_info = "unknown_version"
            try:
                # Attempt to get model version/run_id if available
                if hasattr(MODEL, 'metadata') and MODEL.metadata and hasattr(MODEL.metadata, 'run_id'):
                    model_version_info = MODEL.metadata.run_id
            except Exception:
                pass # Keep default
            
            # Use the numeric prediction for logging
            pred_id = model_monitoring.log_prediction(
                features=feature_row,
                prediction=predictions_numeric[i],
                model_version=model_version_info
            )
            logged_prediction_ids.append(pred_id)
        
        return jsonify({
            "predictions": class_predictions,
            "raw_predictions": predictions_numeric.tolist(),
            "prediction_id": logged_prediction_ids[0] if len(logged_prediction_ids) == 1 else logged_prediction_ids
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint to receive feedback on predictions."""
    try:
        content = request.json
        prediction_id = content['prediction_id']
        actual_class = content['actual_class']
        
        model_monitoring.log_feedback(prediction_id, actual_class)
        retraining_needed = model_monitoring.check_retraining_needed()
        
        return jsonify({
            "success": True,
            "message": "Feedback recorded successfully",
            "retraining_needed": retraining_needed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/monitoring', methods=['GET'])
def monitoring_dashboard():
    """Serve the monitoring dashboard."""
    try:
        report_html = model_monitoring.generate_monitoring_report()
        return report_html
    except FileNotFoundError:
        return "Monitoring report not found. Generate it first.", 404
    except Exception as e:
        return f"Error loading monitoring report: {str(e)}", 500

@app.route('/trigger-retraining', methods=['POST'])
def trigger_retraining_endpoint(): # Renamed to avoid conflict
    """Endpoint to manually trigger model retraining."""
    global MODEL # Allow modification of global MODEL
    try:
        if not model_monitoring.check_retraining_needed():
            return jsonify({
                "success": False,
                "message": "Retraining not recommended by monitoring."
            }), 400 # Bad request if not needed

        print("Manual retraining trigger accepted. Initiating retraining...")
        model_details = model_monitoring.trigger_retraining() # This runs simple_ml_pipeline and model_registry
        print(f"Retraining complete. New model version: {model_details.version}")
        
        # Reload the model in the API
        print("Reloading model in API...")
        load_model_and_store() # This reloads both model and re-initializes store (though store config is static here)
        if MODEL is None:
            return jsonify({"error": "Failed to reload model after retraining."}), 500

        return jsonify({
            "success": True,
            "message": f"Model retrained and API reloaded with model version {model_details.version}"
        })
    except Exception as e:
        print(f"Error during manual retraining trigger: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Retraining trigger failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure Feast store is applied and materialized before running the API for online features.
    # (e.g., run `feast apply` and `feast materialize-incremental <yesterday_iso_date>` in `feature_repository` dir)
    app.run(debug=True, host='0.0.0.0', port=5001) 