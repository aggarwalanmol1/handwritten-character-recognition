# Handwritten Character Recognition using Deep Learning

A deep learning project that recognizes handwritten digits and English alphabets (uppercase and lowercase) using a Convolutional Neural Network (CNN).

## Project Overview

This project uses the EMNIST ByClass dataset to train a CNN model capable of classifying 62 different handwritten character classes.

Users can draw a character using an interactive canvas or upload an image through a Streamlit web application.

## Features

- Recognizes digits (0–9).
- Recognizes uppercase alphabets (A–Z).
- Recognizes lowercase alphabets (a–z).
- CNN-based character classification.
- Interactive drawing canvas.
- Image preprocessing using Pillow.
- Prediction confidence and top 3 predictions.
- Streamlit web interface.

## Technologies Used

- Python
- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- NumPy
- Matplotlib
- Pillow
- Streamlit
- EMNIST ByClass Dataset

## Model Performance

| Metric | Result |
|---|---:|
| Number of Classes | 62 |
| Input Image Size | 28 × 28 |
| Test Accuracy | 87.02% |
| Test Loss | 0.3538 |

## Project Structure

```text
Handwritten-Character-Recognition/
│
├── models/
│   └── character_recognition_v2.keras
│
├── notebooks/
│   └── 01_data_preparation.ipynb
│
├── src/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/handwritten-character-recognition.git
```

2. Navigate to the project directory:

```bash
cd handwritten-character-recognition
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the environment on Windows:

```bash
venv\Scripts\activate
```

5. Install dependencies:

```bash
pip install -r requirements.txt
```

6. Run the application:

```bash
python -m streamlit run src/app.py
```

## Dataset

The project uses the EMNIST ByClass dataset, containing handwritten digits and uppercase and lowercase English letters.

Dataset source: [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset)

## Future Improvements

- Improve recognition accuracy.
- Enhance image preprocessing.
- Support recognition of complete handwritten words.
- Deploy the application online.

## Author

Anmol Aggarwal

B.Tech – Information Technology 