# ECG-Labeling-LSTM
In collaboration with Ron Yang, we created an LSTM machine learning model to label ECG signal arrhythmias as signal as either normal, left bundle branch blockage, right bundle branch blockage, premature ventricle contraction, or atrial premature beat using the TensorFlow library. The data set comes from the open-source MIT-BIH Arrhythmia Database, containing a .csv file for raw ECG signal and a .txt file for annotations.

Database: https://www.kaggle.com/datasets/mondejar/mitbih-database/data

Database information: https://www.physionet.org/content/mitdb/1.0.0/

Previous work has used the CNN model identification but in this code, we will use the LSTM model as it better handles time sequence data as an ECG raw signal would be.


