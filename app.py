from flask import Flask, render_template, request, url_for
import torch
from torchvision import transforms, models
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import random
import glob
from werkzeug.utils import secure_filename
import gdown

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download models if not present
def download_models():
    if not os.path.exists('amy_model.torch'):
        gdown.download('YOUR_GOOGLE_DRIVE_LINK_amy_model.torch', 'amy_model.torch', quiet=False)
    if not os.path.exists('amy_model.tensorflow'):
        gdown.download('YOUR_GOOGLE_DRIVE_LINK_amy_model.tensorflow', 'amy_model.tensorflow', quiet=False)

download_models()

# PyTorch Prediction
def predict_pytorch(image_path, model_path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
    except Exception as e:
        return None, None, None, f"Erreur lors du chargement de l'image : {e}"

    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        return None, None, None, f"Erreur lors du chargement du modèle PyTorch : {e}"
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class = classes[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    probs_dict = {cls: prob.item() * 100 for cls, prob in zip(classes, probabilities)}

    return predicted_class, confidence, probs_dict, None

# TensorFlow Prediction
def predict_tensorflow(image_path, model_path):
    try:
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
    except Exception as e:
        return None, None, None, f"Erreur lors du chargement de l'image : {e}"

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        return None, None, None, f"Erreur lors du chargement du modèle TensorFlow : {e}"

    output = model.predict(image, verbose=0)
    predicted = np.argmax(output, axis=1)
    probabilities = output[0]

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary', 'Autre']
    predicted_class = classes[predicted[0]]
    confidence = probabilities[predicted[0]] * 100
    probs_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}

    return predicted_class, confidence, probs_dict, None

@app.route('/', methods=['GET', 'POST'])
def index():
    tensorflow_available = True
    try:
        import tensorflow
    except ImportError:
        tensorflow_available = False

    pytorch_model_path = 'amy_model.torch'
    tensorflow_model_path = 'amy_model.tensorflow'
    test_dir = 'data/testing'
    image_files = glob.glob(os.path.join(test_dir, '**/*.jpg'), recursive=True)

    # Check prerequisites
    error = None
    if not os.path.exists(pytorch_model_path):
        error = "Modèle PyTorch introuvable."
    if tensorflow_available and not os.path.exists(tensorflow_model_path):
        tensorflow_available = False
    if not os.path.exists(test_dir) or not image_files:
        error = f"Dossier {test_dir} vide ou introuvable."

    if request.method == 'POST':
        framework = request.form.get('framework')
        action = request.form.get('action')

        if action == 'random' and image_files:
            image_path = random.choice(image_files)
            filename = os.path.basename(image_path)
            # Copy random image to uploads folder
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(image_path, 'rb') as src, open(upload_path, 'wb') as dst:
                dst.write(src.read())
        else:
            if 'image' not in request.files:
                return render_template('index.html', error="Aucune image sélectionnée.", tensorflow_available=tensorflow_available)
            file = request.files['image']
            if file.filename == '':
                return render_template('index.html', error="Aucune image sélectionnée.", tensorflow_available=tensorflow_available)
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

        if error:
            return render_template('index.html', error=error, tensorflow_available=tensorflow_available)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if framework == "PyTorch":
            predicted_class, confidence, probabilities, pred_error = predict_pytorch(image_path, pytorch_model_path, device)
        elif framework == "TensorFlow" and tensorflow_available:
            predicted_class, confidence, probabilities, pred_error = predict_tensorflow(image_path, tensorflow_model_path)
        else:
            return render_template('index.html', error="Framework invalide ou TensorFlow non disponible.", tensorflow_available=tensorflow_available)

        if pred_error:
            return render_template('index.html', error=pred_error, tensorflow_available=tensorflow_available)

        return render_template('index.html',
                               predicted_class=predicted_class,
                               confidence=confidence,
                               probabilities=probabilities,
                               framework=framework,
                               image_path=url_for('static', filename=f'uploads/{filename}'),
                               tensorflow_available=tensorflow_available)

    return render_template('index.html', tensorflow_available=tensorflow_available)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
