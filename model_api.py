import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import mlflow.pyfunc

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
            })
            .catch((error) => {
                console.error('Error:', error);
                document.getElementById('prediction').textContent = `Error: ${error}`;
                document.getElementById('result').style.display = 'block';
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
        
        # Return predictions
        return jsonify({
            "predictions": class_predictions,
            "raw_predictions": predictions.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 