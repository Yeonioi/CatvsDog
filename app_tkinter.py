"""
Cat vs Dog Classifier - Tkinter Desktop GUI
Simple desktop application with zero dependencies
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class CatDogClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🐱🐶 Cat vs Dog Classifier")
        self.root.geometry("800x700")
        self.root.resizable(False, False)
        
        # Configure style
        self.root.configure(bg="#f0f0f0")
        
        # Model configuration
        self.IMG_SIZE = 224
        self.MODEL_PATH = 'models/cat_dog_classifier.h5'
        self.UNCERTAIN_MARGIN = 0.15
        
        # Load model
        self.load_model()
        
        # Create UI
        self.create_ui()
        
        # Variables
        self.current_image = None
        self.current_image_path = None
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.MODEL_PATH):
            messagebox.showerror("Error", f"Model not found at {self.MODEL_PATH}\n\nPlease run train.py first.")
            self.root.destroy()
            return
        
        self.model = keras.models.load_model(self.MODEL_PATH)
    
    def create_ui(self):
        """Create the user interface."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Classifier.Horizontal.TProgressbar",
            troughcolor="#dfe6e9",
            background="#16a085",
            bordercolor="#dfe6e9",
            lightcolor="#16a085",
            darkcolor="#16a085"
        )

        # Header
        header_frame = tk.Frame(self.root, bg="#1f2937", height=95)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text="🐱🐶 Cat vs Dog Classifier",
            font=("Segoe UI", 23, "bold"),
            bg="#1f2937",
            fg="white"
        )
        title_label.pack(pady=(12, 4))
        
        subtitle = tk.Label(
            header_frame,
            text="Upload an image and get a prediction with confidence",
            font=("Segoe UI", 10),
            bg="#1f2937",
            fg="#d1d5db"
        )
        subtitle.pack(pady=(0, 10))
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#eef2f7")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Image display area
        self.image_frame = tk.Frame(content_frame, bg="white", relief=tk.FLAT, bd=1, highlightbackground="#d7dde5", highlightthickness=1)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = tk.Label(
            self.image_frame,
            text="📸 No image loaded\n\nClick 'Upload Image' to select a file",
            font=("Segoe UI", 12),
            bg="white",
            fg="#95a5a6"
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Buttons frame
        button_frame = tk.Frame(content_frame, bg="#eef2f7")
        button_frame.pack(fill=tk.X, pady=10)
        
        upload_btn = tk.Button(
            button_frame,
            text="📤 Upload Image",
            command=self.upload_image,
            font=("Segoe UI", 10, "bold"),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=9,
            relief=tk.FLAT,
            cursor="hand2"
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        predict_btn = tk.Button(
            button_frame,
            text="🔍 Predict",
            command=self.predict,
            font=("Segoe UI", 10, "bold"),
            bg="#2ecc71",
            fg="white",
            padx=15,
            pady=9,
            relief=tk.FLAT,
            cursor="hand2",
            state=tk.DISABLED
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        self.predict_btn = predict_btn
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear,
            font=("Segoe UI", 10, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=15,
            pady=9,
            relief=tk.FLAT,
            cursor="hand2"
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        self.result_frame = tk.Frame(content_frame, bg="white", relief=tk.FLAT, bd=1, highlightbackground="#d7dde5", highlightthickness=1)
        self.result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = tk.Label(
            self.result_frame,
            text="Result will appear here",
            font=("Segoe UI", 14, "bold"),
            bg="white",
            fg="#4b5563"
        )
        self.result_label.pack(pady=(12, 4))
        
        self.confidence_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 11),
            bg="white",
            fg="#7f8c8d"
        )
        self.confidence_label.pack(pady=(0, 10))

        self.confidence_bar = ttk.Progressbar(
            self.result_frame,
            style="Classifier.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100,
            length=500
        )
        self.confidence_bar.pack(pady=(0, 14))

        self.message_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 10),
            bg="white",
            fg="#6b7280"
        )
        self.message_label.pack(pady=(0, 12))
        
        # Info frame
        info_frame = tk.Frame(self.root, bg="#e5e7eb", height=60)
        info_frame.pack(fill=tk.X)
        
        info_text = tk.Label(
            info_frame,
            text="💡 Tips: Clear, centered pet photos work best. If confidence is low, the app reports uncertain instead of forcing a class.",
            font=("Segoe UI", 9),
            bg="#e5e7eb",
            fg="#34495e"
        )
        info_text.pack(pady=8)
    
    def upload_image(self):
        """Upload and display an image."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load image
            image = Image.open(file_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.current_image = image
            self.current_image_path = file_path
            
            # Display image (resize for display)
            display_image = image.copy()
            display_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo
            
            # Enable predict button
            self.predict_btn.config(state=tk.NORMAL)
            
            # Clear previous results
            self.result_label.config(text="Result will appear here", fg="#4b5563")
            self.confidence_label.config(text="")
            self.confidence_bar["value"] = 0
            self.message_label.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess image for prediction."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize
        image = image.resize((self.IMG_SIZE, self.IMG_SIZE))
        
        # Convert to array
        image_array = np.array(image).astype(np.float32)
        
        # Normalize
        image_array = image_array / 255.0
        
        # Ensure image shape is always (H, W, 3)
        if len(image_array.shape) == 2:
            image_array = np.stack([image_array] * 3, axis=-1)
        elif len(image_array.shape) == 3 and image_array.shape[-1] == 4:
            image_array = image_array[:, :, :3]

        if len(image_array.shape) != 3 or image_array.shape[-1] != 3:
            raise ValueError("Unsupported image format. Please use a standard JPG or PNG image.")
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        if image_array.shape != (1, self.IMG_SIZE, self.IMG_SIZE, 3):
            raise ValueError("Image preprocessing failed due to invalid tensor shape.")
        
        return image_array
    
    def predict(self):
        """Make prediction on the current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please upload an image first!")
            return
        
        try:
            # Preprocess and predict
            processed_image = self.preprocess_image(self.current_image)
            prediction = self.model.predict(processed_image, verbose=0)[0][0]

            dog_prob = float(prediction)
            cat_prob = 1.0 - dog_prob
            top_prob = max(cat_prob, dog_prob)
            
            # Interpret result
            if abs(prediction - 0.5) <= self.UNCERTAIN_MARGIN:
                label = "⚠️ Neither Cat nor Dog (Uncertain)"
                confidence = top_prob * 100
                color = "#d97706"
                message = "The model is not confident this image is a clear cat or dog. Try another photo."
            elif prediction < 0.5:
                label = "🐱 CAT"
                confidence = cat_prob * 100
                color = "#e74c3c"
                message = "Looks like a cat based on model confidence."
            else:
                label = "🐶 DOG"
                confidence = dog_prob * 100
                color = "#3498db"
                message = "Looks like a dog based on model confidence."
            
            # Update UI
            self.result_label.config(text=label, fg=color)
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1f}%",
                fg="#7f8c8d"
            )
            self.confidence_bar["value"] = confidence
            self.message_label.config(text=message)
            
        except Exception as e:
            messagebox.showerror(
                "Prediction Error",
                "Could not classify this image.\n"
                "Try a clear JPG/PNG photo of a pet.\n\n"
                f"Details: {str(e)}"
            )
    
    def clear(self):
        """Clear the current image and results."""
        self.current_image = None
        self.current_image_path = None
        self.image_label.config(image="", text="📸 No image loaded\n\nClick 'Upload Image' to select a file", fg="#95a5a6")
        self.image_label.image = None
        self.result_label.config(text="Result will appear here", fg="#4b5563")
        self.confidence_label.config(text="")
        self.confidence_bar["value"] = 0
        self.message_label.config(text="")
        self.predict_btn.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = CatDogClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
