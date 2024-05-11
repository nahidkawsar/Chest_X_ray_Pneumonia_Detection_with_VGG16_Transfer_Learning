# Chest X-Ray Pneumonia Detection using VGG16

![Front page](https://github.com/nahidkawsar/Chest_X_ray_Pneumonia_Detection_with_VGG16_Transfer_Learning/assets/149723828/9f6008a3-1715-4dd7-a6f3-b1c9f8deb998)


This project aims to detect pneumonia from chest X-ray images using a convolutional neural network based on the VGG16 architecture. The dataset used in this project can be found on Kaggle [here](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

## Overview

The process involves several key steps:

1. **Download Kaggle Dataset**:

    - #### Set up Kaggle API Token:

    ```bash
    !cp kaggle.json ~/.kaggle/
    ```

    - #### Download the dataset:

    ```bash
    !kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
    ```

2. **Mount Google Drive**:

    - #### Mount Google Drive to save and access files:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Extract Dataset**:

    - #### Extract the downloaded dataset:

    ```python
    import zipfile
    zip_ref = zipfile.ZipFile('/content/chest-xray-pneumonia.zip', 'r')
    zip_ref.extractall('/content')
    zip_ref.close()
    ```

4. **Data Preparation**:

    - #### Load and preprocess the dataset:

    ```python
    train_ds = keras.utils.image_dataset_from_directory(
        directory='/content/chest_xray/train',
        labels='inferred',
        label_mode='int',
        batch_size=32,
        image_size=(150, 150))
    ```

5. **Model Building**:

    - #### Build the VGG16-based model:

    ```python
    conv_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3))

    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ```

6. **Model Training**:

    - #### Train the model:

    ```python
    history = model.fit(train_ds, epochs=10, validation_data=validation_ds)
    ```
 
 ![training](https://github.com/nahidkawsar/Chest_X_ray_Pneumonia_Detection_with_VGG16_Transfer_Learning/assets/149723828/a0da131e-90e6-4bd3-b005-840883a7c934)


7. **Model Evaluation**:

    - #### Evaluate the model's performance:

    ```python
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], color='red', label='train')
    plt.plot(history.history['val_accuracy'], color='blue', label='validation')
    plt.legend()
    plt.show()
    ```

    Training and Validation Accuracy:
![accuracy ](https://github.com/nahidkawsar/Chest_X_ray_Pneumonia_Detection_with_VGG16_Transfer_Learning/assets/149723828/648b79da-efc8-4885-82ed-a37cfe920186)



    ```python
    plt.plot(history.history['loss'], color='red', label='train')
    plt.plot(history.history['val_loss'], color='blue', label='validation')
    plt.legend()
    plt.show()
    ```

    Training and Validation Loss:  
![loss](https://github.com/nahidkawsar/Chest_X_ray_Pneumonia_Detection_with_VGG16_Transfer_Learning/assets/149723828/063b2f67-7484-4fc8-a7a0-ea1cfebe632b)

8. **Model Testing**:

    - #### Test the trained model:

    ```python
    test_input = test_img.reshape((1, 150, 150, 3))
    model.predict(test_input)
    ```

## Getting Started

To replicate this project, follow these steps:

1. Set up your Kaggle API token and clone the repository.
2. Download and extract the dataset.
3. Open the provided Colab notebook to work on the project.
4. Train and evaluate the model, and test it on new images.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV

Ensure you have these dependencies installed using `pip`.

### Author:
H.M Nahid kawsar

Find me in [Linkedin:](linkedin.com/in/h-m-nahid-kawsar-232a86266)
