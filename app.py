from flask import Flask, render_template, request
import os
import cv2

from image_prediction import predict_for_flask

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

# Create folders if not exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = predict_for_flask(filepath)

        # Save labeled image
        result_img_path = os.path.join(RESULT_FOLDER, file.filename)
        cv2.imwrite(result_img_path, result["labeled_image"])

        return render_template(
            "index.html",
            prediction=result["final_class"],
            confidence=round(result["confidence"], 2),
            f1=result["f1_score"],
            accuracy=result["accuracy"],
            svm=result["svm"],
            rf=result["rf"],
            cnn=result["cnn"],
            vgg16=result["vgg16"],
            image_path=result_img_path
        )

    return "Error uploading image"


if __name__ == "__main__":
    app.run(debug=True)