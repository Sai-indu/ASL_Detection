import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("asl_model.h5")

# Prediction function
def predict_image(img_array):
    # Ensure it's normalized and shaped correctly
    prediction = model.predict(img_array)
    return prediction
