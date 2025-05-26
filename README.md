# Facial Recognition - AI 4 570
CV Project for the Computer Vision course at GUtech

## Introduction 
We have implemented a facial recognition workflow using the Haar Cascade classifier for facial detection and the facial_recognition library (which utilises dlib's ResNet-34 encoder architecture). 

## File Overview
- encodings.py - used to generate encodings of images in the [LFW](http://vis-www.cs.umass.edu/lfw/) dataset, along with other images added to the dataset

- encodings.pkl - pickle file containing (encoding, label) pairs for images; generated with encodings.py

- app.py - the Flask app

# Installation
- Clone repo
- Requirements: Python 3.8.0
- Create venv and activate, then run the following commands
```
pip install -r requirements.txt
flask run
```
