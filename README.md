🫁 Lung Cancer Detection using CNN – Streamlit Web App
This is a project designed to detect lung cancer from CT scan images using a Convolutional Neural Network (CNN) model. The application provides a user-friendly web interface built with Streamlit and is capable of making predictions in real-time.

🫁 Lung Cancer Detection using CNN – Streamlit Web App
This is a B.Tech final year project designed to detect lung cancer from CT scan images using a Convolutional Neural Network (CNN) model. The application provides a user-friendly web interface built with Streamlit and is capable of making predictions in real-time.📦 Installation & Setup Guide

🔧 Required Libraries
Make sure you have Python 3.7 or higher. Install all required libraries using:

``pip install -r requirements.txt``

If requirements.txt is not available, manually install the key libraries:

``pip install streamlit numpy pandas pillow tensorflow keras streamlit-option-menu``

🖥️ Software Needed
Python 3.7+
VS Code / PyCharm / Jupyter Notebook
Streamlit
Google Chrome or any modern browser


🚀 How to Run the App
 
1)Clone the repository:
``git clone https://github.com/your-username/lung-cancer-detection
cd lung-cancer-detection ``
 
 2)Run the app using Streamlit:
``streamlit run app.py``

3)Upload a lung CT scan image and receive a prediction instantly.

🤖 About the AI Model
1)The model is a Convolutional Neural Network (CNN) trained on preprocessed CT scan datasets.

2)It uses Keras and TensorFlow as the backend for training and inference.

3)Model file: model.h5 should be saved in the model/ directory.

4)Image preprocessing includes resizing to (224, 224) and normalization.


🔑 API Key Help (Optional – for Advanced Features)
If you plan to integrate voice assistance or external APIs (like ElevenLabs or Groq API for NLP or speech output), you’ll need to:

1)Sign up at the provider (e.g., https://www.elevenlabs.io/)

2)Generate an API Key from your account dashboard

3)Store it securely in your project (e.g., in .env or secrets.toml)

4)Access it in Python:
``import os
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")``
🔐 Never share API keys in public repositories.



🧾 Project Structure

lung-cancer-detection/
│
├── app.py                  # Main Streamlit app
├── model/
│   └── model.h5            # Trained CNN model
├── assets/                 # Icons/images for UI
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation


📊 Dataset Used
You may use open-source CT scan datasets from:

Kaggle Lung CT Dataset

LUNA16 or any annotated DICOM image set


