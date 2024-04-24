
'''
This file runs the validation of the the saved tflite model of ECG signal
classification. Look at the ECG-Labeling-LSTM as reference to the model and signal
processing. Here the code runs tflite-runtime to use the saved model and make
predictions.

It takes in the data and labels for the validation set from the ECG dataset used in the
other code. 

Authors: Zoe Boysen and Ron Yang
'''

#imports
import tflite_runtime as tflite                            # version 2.7.0
import tflite_runtime.interpreter as Interpreter
import numpy as np                                         # version 1.21.0 or at least 1.19.2


#constants
TFLITE_FILE_PATH = '/home/pi/Desktop/2114_model.tflite'
DATACHUNK_FILE_PATH = '/home/pi/Desktop/val_datachunk.csv'
LABEL_FILE_PATH = '/home/pi/Desktop/val_labels.csv'


#get the signal and labels 
val_set_dataset = np.genfromtxt(DATACHUNK_FILE_PATH, delimiter=',')
val_set_labels  = np.genfromtxt(LABEL_FILE_PATH, delimiter=',')
# print(val_set_dataset.shape) #three patients with signals of length 2114
# print(val_set_labels.shape)

patient1_data = val_set_dataset[0][:]
patient1_labels = val_set_labels[0]
# print(patient1_data.shape)
# print(patient1_labels.shape)

#set up the interpreter to read in the tflite model 
interpreter = tflite.interpreter.Interpreter(model_path=TFLITE_FILE_PATH)
interpreter.allocate_tensors()

#get the input and output details of the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\n Input details: ", input_details)
print("\n Output details: ", output_details)

input_shape = input_details[0]['shape']
print("\n Input Shape =" , input_shape)

inp = patient1_data.astype(np.float32)
print(inp.shape)

#resize data based off the input shape found in the model
inp = inp.reshape((input_shape))
print(inp.shape)

interpreter.set_tensor(input_details[0]['index'], inp)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print('\n Output data: ' , output_data)
