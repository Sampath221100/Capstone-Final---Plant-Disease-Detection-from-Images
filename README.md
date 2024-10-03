# Capstone-Final-Plant-Disease-Detection-from-Images

---

# Plant Disease Detection using CNN

## Overview

This project involves the development of a comprehensive web application using **Streamlit** that allows users to upload images of plant leaves and receive predictions about the presence and type of plant disease. The core of this system is a **Convolutional Neural Network (CNN)** model that has been trained on a labeled dataset to classify different plant diseases. The application is designed for real-world agricultural use, helping farmers and gardeners quickly diagnose plant diseases.

## Features

- **Image Upload Interface**: Users can easily upload images of plant leaves for analysis.
- **Real-time Disease Prediction**: The CNN model predicts whether the plant is diseased and specifies the type of disease.
- **Optimized CNN Model**: The model is trained on the **New Plant Diseases Dataset** from Kaggle, ensuring accurate predictions.
- **Pretrained Model Benchmarking**: The performance of the CNN model is compared with at least three pretrained models, ensuring superior performance.
- **User-Friendly Interface**: The application is intuitive and easy to navigate, providing a seamless user experience.
- **Detailed Evaluation Metrics**: The model's performance is assessed using accuracy, precision, and recall.

## Project Structure

```
├── app.py                    # Streamlit application script
├── models/                   # Folder for storing trained models
├── data/                     # Folder for storing dataset (New Plant Diseases Dataset)
├── utils/                    # Helper functions (e.g., image preprocessing)
├── README.md                 # Project documentation
├── requirements.txt          # List of required libraries and dependencies
└── docs/                     # User guide and project report
```

## Key Components

### 1. User Interface Development
- The Streamlit application provides an intuitive interface for uploading images of plant leaves.
- Users receive instant feedback on the uploaded image and predictions about the plant disease.

### 2. Image Preprocessing
- Image preprocessing steps include resizing, normalization, and augmentation to improve model accuracy.
- These steps ensure that the CNN model receives consistent and well-processed input data for better predictions.

### 3. Disease Classification (CNN Model)
- A **Convolutional Neural Network (CNN)** is developed and trained to classify plant diseases.
- The model is trained using the **New Plant Diseases Dataset** from Kaggle, which includes a variety of plant leaf images with labeled diseases.
- Techniques like **data augmentation** and **transfer learning** are used to boost model performance.

### 4. Model Performance and Optimization
- The model is evaluated using metrics such as accuracy, precision, and recall.
- The performance of the CNN model is compared to at least three pretrained models (e.g., ResNet, VGG, Inception) to ensure it outperforms existing solutions.

### 5. Deployment and Testing
- The application is deployed using **Streamlit**, making it easily accessible to end-users.
- Extensive testing is conducted to ensure the application can handle a variety of input images and provide accurate predictions.

## Results

- **Functional Application**: A working web application that allows users to upload plant images and receive predictions in real-time.
- **Model Performance**: Detailed performance reports for the CNN model, including accuracy, precision, and recall metrics.
- **User Guide**: Documentation that explains how to use the application, interpret results, and report issues.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/plant-disease-detection.git
    cd plant-disease-detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

4. Open the application in your browser, upload an image of a plant leaf, and receive the disease prediction.

## Tools and Technologies

- **Programming Language**: Python
- **Framework**: Streamlit for the front-end
- **Libraries**: OpenCV, TensorFlow/Keras or PyTorch for image processing and model development
- **Dataset**: [New Plant Diseases Dataset](https://www.kaggle.com/datasets) from Kaggle

## Future Improvements

- Enhance the model by incorporating more advanced techniques like ensemble learning.
- Add functionality to display more details about the disease and recommended treatments.
- Extend support for mobile applications for easier field diagnosis.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com) for providing the New Plant Diseases Dataset.
- [Streamlit](https://www.streamlit.io) for their powerful application framework.
- Open-source contributions that have made this project possible.

---
