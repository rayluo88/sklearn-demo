import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import mlflow.pyfunc
# Import model monitoring
import model_monitoring

app = Flask(__name__)

# Load the model from the MLflow Model Registry
def load_model(model_name="iris-classifier", stage="Staging"):
    """Load the model from MLflow Model Registry."""
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from {model_uri}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fall back to latest version if stage not found
        model_uri = f"models:/{model_name}/latest"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            print(f"Loaded latest model version instead")
            return model
        except Exception as e:
            print(f"Error loading latest model version: {e}")
            return None

# Global variable for the model
MODEL = load_model()

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
    <title>Iris Classifier API</title>
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
        <h1>Iris Flower Classifier</h1>
        
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
            <p id="confidence"></p>
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
            <p>To use this model programmatically, send a POST request to <code>/predict</code> with the following JSON format:</p>
            <pre>
{
  "features": [
    [sepal_length, sepal_width, petal_length, petal_width]
  ]
}
            </pre>
            
            <h4>Example with curl:</h4>
            <pre>curl -X POST http://localhost:5001/predict \\
  -H "Content-Type: application/json" \\
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
    if MODEL is not None:
        return jsonify({"status": "healthy", "model_loaded": True})
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint.
    
    Expected JSON format:
    {
        "features": [
            [sepal_length, sepal_width, petal_length, petal_width],
            ...
        ]
    }
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        # Parse input data
        content = request.json
        features = np.array(content['features'])
        
        # Make prediction
        predictions = MODEL.predict(features)
        
        # Convert numeric predictions to class names
        class_predictions = [CLASS_MAPPING.get(int(pred), f"Unknown-{pred}") for pred in predictions]
        
        # Log the prediction for monitoring
        prediction_ids = []
        for i, (feature, prediction) in enumerate(zip(features, predictions)):
            # Get model details to log the version
            model_version = "unknown"
            try:
                model_info = MODEL._model_meta.to_dict() if hasattr(MODEL, '_model_meta') else {}
                model_version = model_info.get('run_id', 'unknown')
            except:
                pass
                
            # Log the prediction
            prediction_id = model_monitoring.log_prediction(
                features=feature,
                prediction=prediction,
                model_version=model_version
            )
            prediction_ids.append(prediction_id)
        
        # Return predictions
        return jsonify({
            "predictions": class_predictions,
            "raw_predictions": predictions.tolist(),
            "prediction_id": prediction_ids[0] if len(prediction_ids) == 1 else prediction_ids
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint to receive feedback on predictions.
    
    Expected JSON format:
    {
        "prediction_id": "unique_prediction_id",
        "actual_class": 0|1|2  # The true class
    }
    """
    try:
        content = request.json
        prediction_id = content['prediction_id']
        actual_class = content['actual_class']
        
        # Log the feedback
        model_monitoring.log_feedback(prediction_id, actual_class)
        
        # Check if retraining is needed
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
    # Generate a fresh monitoring report
    model_monitoring.generate_monitoring_report()
    
    # Read the HTML report
    try:
        with open("model_monitoring_report.html", "r") as f:
            report_html = f.read()
        return report_html
    except:
        return "Error loading monitoring report", 500

@app.route('/trigger-retraining', methods=['POST'])
def trigger_retraining():
    """Endpoint to manually trigger model retraining."""
    try:
        # Check if retraining is actually needed
        if not model_monitoring.check_retraining_needed():
            return jsonify({
                "success": False,
                "message": "Retraining not needed based on current metrics"
            }), 400
            
        # Trigger retraining
        model_details = model_monitoring.trigger_retraining()
        
        # Reload the model
        global MODEL
        MODEL = load_model()
        
        return jsonify({
            "success": True,
            "message": f"Model retrained and deployed as version {model_details.version}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 