import torch
from torchvision import transforms, models
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import random
import glob

# Modèle PyTorch (ResNet18, comme dans train_pytorch.py)
def get_pytorch_model():
    model = models.resnet18(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  
    return model

# Prédiction PyTorch
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

    model = get_pytorch_model()
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

    return predicted_class, confidence, probabilities, None

# Prédiction TensorFlow
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

    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    predicted_class = classes[predicted[0]]
    confidence = probabilities[predicted[0]] * 100

    return predicted_class, confidence, probabilities, None

class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Classification des Tumeurs Cérébrales")
        self.root.geometry("800x600")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pytorch_model_path = 'amy_model.torch'
        self.tensorflow_model_path = 'amy_model.tensorflow'
        self.test_dir = 'data/testing'
        self.image_files = glob.glob(os.path.join(self.test_dir, '**/*.jpg'), recursive=True)

        self.tensorflow_available = True
        try:
            import tensorflow
        except ImportError:
            self.tensorflow_available = False

        # Vérifier les prérequis
        self.check_prerequisites()

        # Interface
        self.setup_ui()

    def check_prerequisites(self):
        if not os.path.exists(self.pytorch_model_path):
            messagebox.showerror("Erreur", f"Modèle PyTorch {self.pytorch_model_path} introuvable. Veuillez entraîner avec train_pytorch.py.")
            self.root.quit()
        if self.tensorflow_available and not os.path.exists(self.tensorflow_model_path):
            messagebox.showwarning("Avertissement", f"Modèle TensorFlow {self.tensorflow_model_path} introuvable. TensorFlow désactivé.")
            self.tensorflow_available = False
        if not os.path.exists(self.test_dir) or not self.image_files:
            messagebox.showerror("Erreur", f"Dossier {self.test_dir} vide ou introuvable. Veuillez exécuter utils/prep_pytorch.py.")
            self.root.quit()

    def setup_ui(self):
        # Cadre principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Titre
        ttk.Label(main_frame, text="Classification des Tumeurs Cérébrales", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

        # Choix du framework
        ttk.Label(main_frame, text="Framework :").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.framework_var = tk.StringVar(value="PyTorch")
        framework_menu = ttk.Combobox(main_frame, textvariable=self.framework_var, state="readonly")
        framework_menu['values'] = ["PyTorch"] if not self.tensorflow_available else ["PyTorch", "TensorFlow"]
        framework_menu.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Bouton pour sélectionner une image
        ttk.Button(main_frame, text="Choisir une image", command=self.select_image).grid(row=2, column=0, columnspan=2, pady=10)

        # Bouton pour image aléatoire
        ttk.Button(main_frame, text="Image aléatoire", command=self.random_image).grid(row=3, column=0, columnspan=2, pady=10)

        # Affichage de l'image
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Résultats
        self.result_text = tk.Text(main_frame, height=6, width=50, state='disabled')
        self.result_text.grid(row=5, column=0, columnspan=2, pady=10)

        # Bouton quitter
        ttk.Button(main_frame, text="Quitter", command=self.root.quit).grid(row=6, column=0, columnspan=2, pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if file_path:
            self.process_image(file_path)

    def random_image(self):
        if self.image_files:
            file_path = random.choice(self.image_files)
            self.process_image(file_path)
        else:
            messagebox.showerror("Erreur", "Aucune image disponible dans data/testing.")

    def process_image(self, image_path):
        # Afficher l'image
        try:
            img = Image.open(image_path).resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher l'image : {e}")
            return

        # Faire la prédiction
        framework = self.framework_var.get()
        if framework == "PyTorch":
            predicted_class, confidence, probabilities, error = predict_pytorch(image_path, self.pytorch_model_path, self.device)
        else:
            predicted_class, confidence, probabilities, error = predict_tensorflow(image_path, self.tensorflow_model_path)

        if error:
            messagebox.showerror("Erreur", error)
            return

        # Afficher les résultats
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Prédiction {framework} : {predicted_class} (Confiance : {confidence:.2f}%)\n")
        self.result_text.insert(tk.END, "Probabilités :\n")
        probs = probabilities.numpy() if framework == "PyTorch" else probabilities
        for cls, prob in zip(['glioma', 'meningioma', 'notumor', 'pituitary'], probs):
            self.result_text.insert(tk.END, f"  {cls}: {prob*100:.2f}%\n")
        self.result_text.config(state='disabled')

def main():
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()