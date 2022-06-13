# ProjectNom-streamlit
Streamlit webapps for Project Nom. Developmental purpose for model deployment. This repository contains the code for testing machine learning model.

---

Table of Contents
1. About the project
2. Machine Learning documentation

---

## About the Project

To achieves a healthy life and dream body, a neatly structured meal tracking is important to keep track of fulfilled nutrition. We present an android application used to classify food items from images taken by users from their own mobile devices, return its prediction result and the nutrition facts, and log it into the applications. Now, users can neatly structured their meal everyday without hassle.

---

## Machine Learning Documentations

The model developed using Google Colaboratory environtments. Open and create a copy for the `ipynb` file notebook or download into your local computer to use you own machines.

Link to google colaboratory notebook: https://colab.research.google.com/drive/15SHjLsFI5R1neUdT5uUaJvkGwDWEljPS?usp=sharing

1. Load the Dataset
    - Dataset used for this project is Food-101. You can find the dataset here: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
2. Preprocess the dataset for model input
    - We're using `ImageDataGenerator` to help us structure our training and validation set.
    - Splitting the dataset into 0.8 training and 0.2 validation
    - Resizing the dataset to 299 x 299
3. Train the Model
    - We're using transfer learning with InceptionV3 model for better accuracy. InceptionV3 layers freezed to prevent from re-training the layers.
    - Connecting the InceptionV3 output with our custom architecture to serve our case.
    - Adding several layers to our custom architecture:
        - Added MaxPooling2D layer for edge detection
        - Added Dense layer with 512 units and relu activation
        - Added Dropout layer with 0.2 rates
        - Added a Flatten layer
        - Added an output layer, which is Dense layer with 101 units, L2 kernel regularizers with rates of 5e-3, and softmax activation
    - Optimizers using `Adam` with rates of 1e-4
    - Losses using `CategoricalCrossentropy`
    - Setting up `ModelCheckpoint`, `CSVLog`, and `Tensorboard` callbacks to help developing the model
    - Trained with 30 epochs
4. Result
    - Our final result after training with 30 epochs as follow:
        - `loss: 1.4992`
        - `acc: 0.6486`
        - `val_loss: 1.5558`
        - `val_acc: 0.6461`

