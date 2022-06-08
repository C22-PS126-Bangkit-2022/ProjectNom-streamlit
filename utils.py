# Python library
import io
import requests
from PIL import Image

# ML library
import numpy as np
import tensorflow as tf
import streamlit as st

# Google Cloud Library
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions

#MODEL = tf.keras.models.load_model("savedmodel_icv3_11class", compile=False)
MODEL = tf.keras.models.load_model(
    "model/checkpoint_icv3.hdf5", compile=False)
LABELS_11CLASS = ['apple_pie', 'beef_carpaccio', 'bibimbap', 'cup_cakes', 'foie_gras',
                  'french_fries', 'garlic_bread', 'pizza', 'spring_rolls', 'spaghetti_carbonara',
                  'strawberry_shortcake']

preprocess_label_dict = {
    "beignets": "",
    "beef_carpaccio": "carpaccio",
    "croque_madame": "",
    "fried_calamari": "calamari",
    "takoyaki": "dumplings",
    "spaghetti_carbonara": "carbonara",
    "lobster_roll_sandwich": "sandwich",
}

with open("./classes.txt") as file:
    lines = file.readlines()
    LABELS = [line.rstrip() for line in lines]

model_options = {
    "model_1": {
        "classes": LABELS,
        "model_name": MODEL  # change to be your model name
    },
    "model_2": {
        "classes": LABELS_11CLASS,
        "model_name": "nom_11classes_DELETELATER"
    },
    "model_3": {
        "classes": LABELS_11CLASS,
        "model_name": "nom_11classes_DELETELATER"
    }
}


# def preprocess_image(filename, img_shape=299, rescale=True):
#     """
#     Function to preprocessing the uploaded image.
#     """
#     img = tf.keras.preprocessing.image.load_img(input_image, target_size=(299, 299))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     # img = np.expand_dims(img, axis=0)

#     # img = tf.io.decode_image(filename, channels=3)
#     # img = tf.image.resize(img, [img_shape, img_shape])

#     if rescale:
#         return img/255.
#     else:
#         return img

# def preprocess_image(filename, img_shape=299, rescale=True):
#     """
#     Function to preprocessing the uploaded image.
#     La kok error semua le
#     """
#     img = tf.keras.preprocessing.image.load_img(
#         filename, target_size=(img_shape, img_shape))
#     img = tf.keras.preprocessing.image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)

#     if rescale:
#         return img/255.
#     else:
#         return img

def preprocess_image(filename, img_shape=299, rescale=True):
    """
    Function to preprocessing the uploaded image.
    """
    pil_image = Image.open(io.BytesIO(filename)).resize(
        (img_shape, img_shape), Image.LANCZOS).convert('RGB')
    image = np.asarray(pil_image)
    batch = np.expand_dims(image, axis=0)

    return batch / 255.0


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)

    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options, cache_discovery=False)  # cache_discovery needed to bypass oauth2client unavailable error
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    instances_list = instances.tolist()
    response = service.projects().predict(
        name=name,
        body={'instances': instances_list}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


# @st.cache
# def run_inference(image, model, label_list):
#     """
#     Function to run prediction with uploaded image
#     """
#     image = preprocess_image(image)
#     result = model.predict(image)

#     # Get the highest probabilities classes, then fetch the label name
#     result_class = label_list[np.argmax(result)]
#     result_confidence = np.amax(result)

#     return image, result_class, result_confidence


def get_nutrition(api_key, prediction):
    if preprocess_label_dict.__contains__(prediction):
        prediction = preprocess_label_dict[prediction]
        print("Prediction result: ", prediction)

    query = prediction.replace("_", " ")
    api_url = 'https://api.calorieninjas.com/v1/nutrition?query={}'.format(
        query)
    print("API URL", api_url)

    response = requests.get(api_url, headers={'X-Api-Key': api_key})
    print("Response: ", response)

    if response.status_code == requests.codes.ok:
        return response.text
    else:
        # print("Error: ", response.status_codes, response.text)
        return f"Error: {response.status_codes}."
