# Arsal Arabic Sign Language Recognition

An AI-powered Arabic Sign Language recognition project containing two deployed applications:

1. **Arabic Sign Language Letters Recognition**
2. **Arabic Sign Language Gesture Recognition**

The project uses Computer Vision, MediaPipe, Machine Learning, and Deep Learning techniques to recognize Arabic sign language from images and videos.

---

## Live Applications

### Arabic Sign Language Letters

Recognizes Arabic sign language letters from hand images.

**Features**

* Image upload support
* Hand landmark extraction using MediaPipe
* Machine Learning classification
* Instant prediction results

**Demo URL**

```
https://abdelrhman632--arsal-alphabet-letters-gradio-app.modal.run/
```

---

### Arabic Sign Language Gestures

Recognizes Arabic sign language gestures from uploaded videos.

**Features**

* Video upload support
* Landmark extraction using MediaPipe Holistic
* LSTM-based gesture recognition
* Confidence score reporting

**Demo URL**

```
https://abdelrhman632--arsal-gestures-gradio-app.modal.run/
```

---

## Repository Structure

```text
Arsal-alphabet/
│
├── Letters/
│   ├── app.py
│   ├── modal_app.py
│   ├── arasl_alphabet_model_2.pkl
│   ├── requirements.txt
│   └── ...
│
├── Gestures/
│   ├── app.py
│   ├── modal_app.py
│   ├── arabic_sign_lstm_model.h5
│   ├── label_encoder.pkl
│   ├── holistic_landmarker.task
│   ├── requirements.txt
│   └── ...
│
└── README.md
```

---

## Technologies Used

* Python
* Gradio
* MediaPipe
* TensorFlow / Keras
* Scikit-Learn
* OpenCV
* Modal
* Git LFS

---

## Deployment

Both applications are deployed using Modal.

### Deploy Letters Application

```bash
modal deploy Letters/modal_app.py
```

### Deploy Gestures Application

```bash
modal deploy Gestures/modal_app.py
```

---

## Local Installation

Clone the repository:

```bash
git clone https://github.com/abdelrhman632/Arsal-alphabet.git
cd Arsal-alphabet
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r Letters/requirements.txt
```

or

```bash
pip install -r Gestures/requirements.txt
```

---

## Team
## Development Team

This project was developed by:

* Abdelrahman Yasser
* Rowida Ahmed
* Omar Abdelaziz
* Mazen Moataz
* Roaa Ahmed

The team collaborated on dataset preparation, model development, computer vision processing, deployment, and user interface development for Arabic Sign Language recognition.


Project focused on Arabic Sign Language recognition using Computer Vision and Deep Learning techniques to improve accessibility and communication support for Arabic-speaking sign language users.

---

## License

This project is intended for educational and research purposes.
