'''
This file runs the validation of the the saved tflite model of ECG signal
classification. Look at the ECG-Labeling-LSTM as reference to the model and signal
processing. Here the code runs tflite-runtime to use the saved model and make
predictions.

Using the neurokit2 library we simulate chunks of ECG signal for a 'live' labeling.

Authors: Zoe Boysen and Ron Yang
'''

#import
import tflite_runtime as tflite                            # version 2.7.0
import tflite_runtime.interpreter as Interpreter
import numpy as np                                         # version 1.21.0
#import pywt                                               # PyWavelets version 1.3.0
import neurokit2 as nk                                     # version #version 2.7.0


#constants 
TFLITE_FILE_PATH = '/home/pi/Desktop/better_Carlos.tflite'

#get the signal and labels 
#val_set_dataset = np.genfromtxt('/home/pi/Desktop/val_datachunk.csv', delimiter=',')
#val_set_labels  = np.genfromtxt('/home/pi/Desktop/val_labels.csv', delimiter=',')

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


#inp = val_set_dataset.astype(np.float32)
#resize data based off the input shape found in the model
#loop 900 times for about simulated 30 min
inp=[]
for i in range(32):
    inp.append(nk.ecg_simulate(duration = 2, sampling_rate = 362, heart_rate=80, random_state=i))
#     inp = np.array(inp).astype(np.float32)
#     inp = inp.reshape((32,724,1))
#     interpreter.set_tensor(input_details[0]['index'], inp)
#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     print('\n Output data: ' , output_data)

inp = np.array(inp).astype(np.float32)
#resize data based off the input shape found in the model
inp = inp.reshape((32,724,1))
print(inp.shape)

# interpreter.set_tensor(input_details[0]['index'], inp)
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print('\n Output data: ' , output_data)

