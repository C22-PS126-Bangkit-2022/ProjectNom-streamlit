import os
from requests import session
# from main import get_nutrition

import numpy as np
import tensorflow as tf
import streamlit as st
import SessionState

from utils import preprocess_image, get_nutrition, predict_json, model_options

# Load the model and the labels
# with open('classes.txt') as file:
#     lines = file.readlines()
#     LABELS = [line.rstrip() for line in lines]

# Setup Cloud Environment Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'project-nom-351311-a6e5d5ac5601.json'
PROJECT = "Project Nom"
REGION = "asia-southeast2"

# DEVELOPMENT PURPOSE. REMEMBER TO DELETE LATER
API_KEY = 'eN4R4KzaBHkdU1L0hTsqqCWpGL20kFWWMLNlpTSS'


@st.cache
def run_inference(image, model, label_list):
    """
    Function to run prediction with uploaded image
    """
    image = preprocess_image(image)
    # result = predict_json(
    #     project=PROJECT,
    #     region=REGION,
    #     model=model,
    #     instances=image
    # )
    result = model.predict(image)

    # Get the highest probabilities classes, then fetch the label name
    result_class = label_list[np.argmax(result)]
    result_confidence = np.amax(result)

    return image, result_class, result_confidence


# Pick the model version
# choose_model = st.sidebar.selectbox(
#     "Pick model you'd like to use",

#     ("Full Model (All classes)",  # original
#      "Alternative Model 1 (11 food classes)",  # original 10 classes + donuts
#      "Alternative Model 2 (11 food classes + non-food class)")  # 11 classes (same as above) + not_food class
# )

# Sidebar
model_version = st.sidebar.selectbox(
    "Pick model you'd like to use",

    ("Full Model (All classes)",  # original
     "Alternative Model 1 (11 food classes)",  # original 10 classes + donuts
     "Alternative Model 2 (11 food classes + non-food class)")  # 11 classes (same as above) + not_food class
)

# Model choice logic
if model_version == "Full Model (All classes)":
    LABELS = model_options["model_1"]["classes"]
    MODEL = model_options["model_1"]["model_name"]
elif model_version == "Alternative Model 1 (11 food classes)":
    LABELS = model_options["model_2"]["classes"]
    MODEL = model_options["model_2"]["model_name"]
else:
    LABELS = model_options["model_3"]["classes"]
    MODEL = model_options["model_3"]["model_name"]

# Display info about model and classes
if st.checkbox("Show classes"):
    st.write(
        f"You chose {MODEL}, these are the classes of food it can identify:\n", LABELS)

# Page title
st.title("Nom Development Page")
st.header("Page to test the ML model before integrating with Android")


upload_img = st.file_uploader(
    label="Upload your food image (Supported file type: png, jpg, jpeg)",
    type=['png', 'jpg', 'jpeg']
)

# Setting up SessionState. #
session_state = SessionState.get(pred_button=False)

# Logic of app flow
if not upload_img:
    st.warning("Insert an image.")
    st.stop()
else:
    # If success, read the image
    session_state.uploaded_image = upload_img.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# If predict button clicked
if pred_button:
    session_state.pred_button = True

if session_state.pred_button:
    session_state.image, session_state.result_class, session_state.result_confidence = run_inference(
        session_state.uploaded_image, model=MODEL, label_list=LABELS)

    session_state.nutrition_facts = get_nutrition(
        API_KEY, session_state.result_class)

    st.write(f"Prediction: {session_state.result_class} \
      Confidence: {session_state.result_confidence:.3f}")

    st.write(f"Nutrition Facts: {session_state.nutrition_facts}")

# Next: Continuing app.py
