from flask import Flask, render_template, request, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("nail_disease_model.h5")

# ✅ CLASS LABELS (must match training)
class_labels = [
    "Acral_Lentiginous_Melanoma",  # 0
    "Healthy_Nail",                # 1
    "Onychogryphosis",             # 2
    "blue_finger",                 # 3
    "Clubbing",                    # 4
    "pitting"                      # 5
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No selected file")

        if file:
            # Save image
            save_path = os.path.join("static", file.filename)
            file.save(save_path)

            img_path = file.filename  # only filename for HTML

            # Preprocess image
            img = image.load_img(save_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            pred = model.predict(img_array)
            class_index = int(np.argmax(pred[0]))

            # ✅ Safety check for out of range
            if class_index >= len(class_labels):
                prediction = "Unknown"
                confidence = 0
            else:
                prediction = class_labels[class_index]
                confidence = round(float(pred[0][class_index]) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path
    )

if __name__ == "__main__":
    app.run(debug=True)
