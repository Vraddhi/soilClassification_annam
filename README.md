# soilClassification_annam
This repo consists of the assignment tasks given by annam in the preliminary part of the hackathon 


Prerequisites
Make sure you have the following installed:
Python 3.8+
pip
Virtual environment tool (optional but recommended)

 Install Dependencies
pip install -r requirements.txt
Ensure that requirements.txt contains:
tensorflow, scikit-learn, pandas, numpy, matplotlib, opencv-python, tqdm, seaborn


Instructions to Run the Project
 Challenge 1 – 4-Class Soil Type Classification
To Train and Predict
Run the Notebook this will train the classification model using CNN + data augmentation.

Output submission.csv with:
image_id,label
Include preprocessing, training, prediction, and CSV generation.



Challenge 2 – Soil vs Not Soil Classification
To Train and Predict
Run the Notebook this will train the autoencoder + anomaly detection model.

Generate submission.csv in the same folder with:
image_id,label
