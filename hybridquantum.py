import pennylane as qml #quantum machine learning library. provides tools to create quantum circuits that work with machine learning. PennyLane bridges quantum computing frameworks (like Qiskit) with ML frameworks (like PyTorch)
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# create quantum device simulator
n_qbits = 4 # 4 qubits can represent 2^4 = 16 different quantum states simultaneously
dev = qml.device('default.qubit', wires=n_qbits) # This is our device. 'default.qubit' is PennyLane's built-in quantum simulator. Wires=nqubits tells the device how many qubits to prepare.

@qml.qnode(dev) #python decorator, transforms the quantum function into a node. Runs the node on qml.device 'default.qbit'
def quantum_circuit(inputs, weights): # inputs = classical data (0.5) weights = trainable parameters (weights in a neural network)
    for i in range(n_qbits): # loops through each bit and applys the operation individually.
        qml.RY(inputs[i], wires=i) # RY is rotation gate around the Y-axis, inputs[i] represents the angle of rotation and wires=i specifies which qbits to rotate. 
            # ^ encodes classical data information into quantum states ^ 

    # adding trainable quatum operations
    for layer in range(2): # creates 2 layers of trainable quantum operations.
            for i in range(n_qbits): # Loop through each qbit again. (0, 1, 2, 3)
                qml.RX(weights[layer, i, 0], wires=i) # Rotation around the x axis of the qbit with trainable parameters via gradient descent
                qml.RZ(weights[layer, i, 1], wires=i) # Rotation around the z axis of the qbit with different trainable parameter. each qbit has 2 trainable parameters per angle. (RX & RZ rotations)

                    # Weights shape = 2 layers, 4 qbits, 2 rotations. 16 quantum parameters. (2 x 4 x 2 = 16)

            # Adding entanglment between adjacent qbits
            for i in range(n_qbits - 1): #Loop through qbits 0, 1 and 2 but not 3. 
                qml.CNOT(wires=[i, i+1]) # This is a controlled-NOT gate (fundamental in quantum). This gate creates entanglement between two qbits. Wires connects qbit i to i+1. 

                    # ^ We get 0<>1, 1<>2, 2<>3 connections. If the first qubit (control) is in state |0, nothing happens to the secound. If first qbit is in state 1, it flips the second qubit. 

                    # Encode classical data with quantum rotations (RY gates)
                    # Process with trainable rotations (RX, RZ gates) - 2 layers
                    # Entangle qubits to create quantum correlations (CNOT gates)

                    #Qubit 0: ─RY─RX─RZ─●─RX─RZ─●─
                    #Qubit 1: ─RY─RX─RZ─X─●─RX─RZ─X─●─
                    #Qubit 2: ─RY─RX─RZ───X─●─RX─RZ─X─
                    #Qubit 3: ─RY─RX─RZ─────X───RX─RZ─

        # Measure quantum states to get the classical outputs. 
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qbits)] # qml.PauliZ is a quantum measurement operator. expval is expect value or average measured result. wires=i measures qubit number i. Returns a number between -1 and +!

            # This loop measures all 4 qubits indepentantly using the PauliZ operator (qml.PauliZ(wires=i)) so that each one gives us a classical number. 4 outputs of classical numbers. 
            # The quantum circuit acts like a quantum layer in our hybrid model. 
            # PauliZ means +1: qubit is in state |0, (quantum up)
            # -1 qubit is in state |1, (quantum down)
            # Between -1 and +1: qubit is in superposition. 
            # The closer we get to >+ 1, the more "classical the state."
            # expval gives us the average result over many measurements. Creates a smooths differentiable output. Great for gradient based computing. 

            # TensorFlow model that combines classical and quantum layers:

class HybridQuantumModel(keras.Model):  # Note: keras.Model, not Keras.Model
    def __init__(self, quantum_circuit, weight_shapes):
        super().__init__()
        
            # Classical preprocessing layer. takes any input size > 4 quantum inputs

        self.pre_layer = keras.layers.Dense(n_qbits, activation='tanh')
            # ^ create an attribute that blongs to our model class. self means this specific model instance. We can access this layer later as self.pre_layer. "Dense" fully connected classical neural entwork layer. 
            # every input connects to every output with learnable weights. Most common layer in ML. "nqubits" the number of outputs this layer produces. Need 4 outputs because we have 4 qubits to feed information to. 
            # tanh is the hyperbolic tanget activation function. any real number -x to +x. Only outputs numbers -1 to +1 with a smooth s-curve. Tanh is best for bounded angles and -1, +1 output is perfect for quantum rotations. Symmetrical around point 0 this is good for quantum encoding
            # Smooth gradients are good for training. 

            # Quantum Layer
        self.quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=n_qbits)
            # ^ Creates another model attribute to store quantum layer inside of. performs quantum processing. using self.quantum, it becomes a part of the models architecture. 
            # qml.qnn is PennyLanes neural network submodel and kerasLayer is a special wrapper that adapts quantum circuits to TensorFLow. It pulls our quantum_curcuit function from earlier (RY, RZ, RX and CNOT gates) while KerasLayer executes it tduring the training & prediction sessions. 
            # Circuit becomes trainable layer identical to Dense or Conv2d. Weight shapes tells TensorFlowhow to organize parameters. Defines shape and structure of trainable quantum weights. Also descripes the dimensions of our quantum parameter array. 
            # 'output=dim-n_qbits' ensures the layer is outputting 4 values (per qubit measured)

            # Classical Output layer. 
        self.post_layer = keras.layers.Dense(1, activation='sigmoid')
            # self.post_layer is the final layer of the hybrid model. Ultimate decision maker  for the prediction. Takes quantum processed features and converts them to a final answer. 
            # keras.layers.Dense(1, is a connected layer with only one output. intellegently combines the 4 quantum measurments and learns how to interpret the circuits output for the problem at hand. 
            # sigmoid activation has an input range of -x - +x and an output range of 0 - 1 with an S shape curve that approaches 0 and 1 asymptotically.

    def call(self, inputs):
            # takes raw input data and psses it through the first layer of the machine. 
        
        x = self.pre_layer(inputs) 
            # calls the layer earlier defined that uses the tanh activation function. 
            # Inputs can be any shape, it must match the model. 
            # x will be whatever we predefined erlier. 
            # Each value is between -1 and +1 because of tanh.

        x = self.quantum_layer(x) 
            # takes the 4 classical values from the pre_layer and sends the to the quantum circuit.
            # self.quantum_layer is the qml.qnnKerasLayer defined earlier.
            # x (input) contains 4 values between -1 and +1
            # x (output) contains 4 quantum-processed values between -1 and +1
        output = self.post_layer(x)
        return output

weight_shapes = {"weights": (2, n_qbits, 2)} 
    # create python dictionary that describes parameter organization
    # PennyLanes KerasLayer expects this specific format
    # must be defined before we create our model instance.
    # Weights is a parametername that matches weights in our quantum_circuit function. 
X, y = make_classification(n_samples=1000, n_features=4, n_redundant=0, n_informative=4, n_classes=2, random_state=42)
    # X.shape = (1000, 4) 1000 examples ech with 4 features aka qbits. 
    # Y.SHAPE = 1000, 1000 LABELS  0 or 1.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # train_test_split # native sklearn functions that randomly splits dataset, take the full x, y dataset and divideds ensuring training and test data have simmilar class distrobutions. 
    # .02 test size reserves 20% of the testing material for testing examples.
    # This lkeaves 80% for training. 
    # 200 tesing data points and 800 training data points. 
    # Models use the training set to learn from the dataset. It then uses the tesing set to evaluate the performance of unseen data. 
    # Splitting data prevents overfitting and memorization of a certain data set, allowing the model to predict outcomes on different types of data, not just the data presented to the model during training. 
model = HybridQuantumModel(quantum_circuit, weight_shapes)
    # create a variable to store our working model inside of. image createing a spcific car from a blueprint. 
    # the HybridQuantumModel it calls the constructor aka init methos of the custom class. 
    # ^ provides and instance fo all the layers presented and defined. (pre_layer, quantum_layer, post_layer)
    # quantum_circuit, psses quantum funtion to the model. KerasLayer will use it to create quantum processing.its the bridge of our quantum circuit definition to the actual model. 
    # weight shapes define the shape of our model, how to organize the 16 trainable parameters. 2, 4, 2 parameter array for quantum rotations. 
    # initializes the quantum weights for the rotations to take place. Allocates memory for the forward and backward passes to take place. 
model.compile( # model configuration for training. needs to be initialized before training can begin. It also sets up the learning parameters. 
     optimizer='adam', # the adam algorythm is one of many that adjusts weights during training. usues momenttum and adaptive learning rates. Usually works out of the box with no tuning. alt optimizers are sgd, rmsprop, adagrad
     loss='binary_crossentropy', # A loss funtion to measure how wrong the predictions are. This is a standard loss function for binary classification ( 0 or 1). Adds more penalty to confident wrong predictions and unceartain. Tries to minimize loss during training. 
     metrics=['accuracy'] # add "accuracy" to determine the percentage of correct predictions. 
)
    # model.compile ^ links the optimizer (adam) to the classical and quantum weights, sets up automatic differentiation and enables gradient computation for both quantum and classical components while tracking loss computation and metrics

    # Train the model
print("Training Quantum Machine Learning model")
history = model.fit( # fit is TensorFlows training method. Thistory tracks loss/accuracy over time and contains all training metrics.  
    X_train, y_train, # training data with 4 features each. Models learns the pattens with this data in order to produce weights. 
    batch_size=32 # processes 32 examples before updating weights. smaller batches usually = better learning.  
    epochs=20 # The number of times the model will train on the given dataset. Looking for training performance to imporve with each repeating epoch.  
    validation_split=0.2 # determines the amount of data used for validtion and training. 0.02 means 20% of the data will be used in the validation split portion.  
    verbose=1 # shows training progress per training epoch.  
) 
 
# Evaluating the performance of our model. 
print("\nEvaluating model...") 
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) # 
print(f"Test Accuracy: {test_accuracy:.4}") 

predictions = model.predict(X_test[:10]) # predict first 10 test samples, returning probability predictions. 
print(f"\nSample predictions: {predictions.flatten()}") # Gives a cleaner output and converts the shape from (10, 0) to 10. 
print(f"Actual labels: {y_test[:10]}") # Shows true answers for 10 examples. 

plt.figure(figsize-(12, 4)) # creates a new plot and adds the parameters. 

plt.subplot(1, 2, 1) # Create a subplot grid with 1 row and 2 columns while activating the first subplot, making room for side by side plots. 
plt.plot(history.history['loss'], label='Training Loss') # plots the loss of each epoch
plt.plot(history.history['val_loss'], label='Validation Loss') # plot validation loss on same axes
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("\nHybrid quantum machine learning model training complete!")

