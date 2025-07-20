# Dostoevsky Text Generator

This project is a web application that uses a custom-built Transformer model, trained from scratch, to generate text in the style of the Russian novelist Fyodor Dostoevsky. The backend is a Flask API that serves the trained model, and the frontend is a simple HTML/CSS/JS interface that allows users to interact with it.




## Features

- **Custom Transformer Model:** A custom built decoder-only Transformer built with TensorFlow and Keras.
- **Custom Tokenizer:** A Byte-Pair Encoding (BPE) tokenizer trained on Dostoevsky's writings.
- **Flask API Backend:** A simple and efficient backend to serve the model and handle generation requests.
- **Saved Generated Sample:** I have saved a generated sample in the assets directory . It has 5000 characters .

---

## Technology Stack

- **Backend:** Python, Flask, TensorFlow
- **Frontend:** HTML, CSS, JavaScript
- **ML Model** Pyhton , Numpy , Tensorflow

---

## Setup and Installation

To run this project locally, follow these steps:

### 1. Clone the Repository

```bash
git clone [https://github.com/AshishRaj04/dostoevsky-generator.git](https://github.com/AshishRaj04/dostoevsky-generator.git)
cd dostoevsky-generator
```
### 2. Set Up a Virtual Environment
```bash
python -m venv venv
```
### 3. Activate it
```bash
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies
```bash
# Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt
```
### 5. Place Model Files
You will need the trained model weights and tokenizer file. Place them in the correct directories:

Place dostoevsky_model.weights.h5 inside the /checkpoints directory.

Place tokenizer.pkl inside the /engine directory.

### 6. Run the Flask Application
Start the backend server by running app.py.
```bash
python app.py
```
The API will now be running at http://localhost:8000.

### 7. Open the Frontend
Simply open the index.html file in your web browser to interact with the application.

API Endpoint
The application exposes a single API endpoint for text generation.

URL: /api/generate

Method: POST

Description: Triggers the model to generate a new text sample.

Success Response (200):
```bash
{
  "status": "success",
  "generated_text": "..."
}
Error Response (500):

{
  "status": "error",
  "message": "Error description..."
}
```