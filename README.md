# Plant-Disease-Detection-System
  A deep learning-based web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN).    The system is built with TensorFlow/Keras  for the model and Streamlit for the interactive web interface.

The system allows users to:
- Upload plant leaf images
- Analyze them using a trained deep learning model
- Get instant disease predictions

#  Model Details

- Framework: TensorFlow / Keras  
- Model Type: Convolutional Neural Network (CNN)  
- Input Size: 128 × 128 images  
- Output Classes: 38 plant disease categories  
- Dataset Split:
  - Training: ~70,295 images  
  - Validation: ~17,572 images  
  - Testing: 33 images

# Model Training
The model was trained using the Jupyter Notebook:
  train_plant_disease.ipynb

Steps include:
* Data preprocessing and augmentation
* CNN architecture design
* Model training using categorical classification
* Evaluation on validation set
* Saving trained model as .keras

# Features
* Upload plant leaf images
* Real-time disease prediction
* 38-class classification system
* Simple and user-friendly UI
* Fast inference using trained CNN model
* Lightweight Streamlit web app

# Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* PIL (Image Processing)

## Setup and Installation

Follow these steps to configure your environment and run the Streamlit application locally on your machine.

### Prerequisites
Make sure you have [Anaconda](https://anaconda.com) installed on your system.

### 1. Environment Setup
Open your terminal and activate the project environment. If you created a dedicated environment for training your model, activate it here:

```bash
conda activate tensorflow
```

### 2. Install Dependencies
Ensure all the required packages for running the web application and the model inference are installed:

```bash
pip install streamlit tensorflow scikit-learn opencv-python matplotlib
```

### 3. Running the Application
Navigate to your project root folder (`streamlit-demo`) where `main.py` is located, and launch the server:

```bash
streamlit run main.py
```

After running this command, your default browser should automatically open a new tab at `http://localhost:8501` displaying the app interface. If it doesn't open automatically, you can copy and paste that local URL manually.


  
