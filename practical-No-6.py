"""
#Practical 6)----------
#AIM: Implement feed forward back propagation neural network learning algorithm.

"""
'''
The process of adding more layers to a neural network-- called deep learning.
The main difference in the code from the single-layer neural net is that
the two layers influence the calculations for the error, and therefore
the adjustment of weights. The errors from the second layer of neurons need
to be propagated backwards to the first layer, this is called backpropagation.
'''
import numpy as np

class NeuralNetwork():
    def __init__(self):
        #seeding for random number generation
        # Seed random number generator, so it generates the same number
        # every time program runs
        np.random.seed()

        #converting weights to a 3 by 1 matrix
        # Model single neuron, with 3 input connections and 1 output connection
        # Assign random weights to a 3x1 matrix, with values in range -1 to 1
        # and mean of 0
        '''
        np.random.rand(3,2)
array([[0.98653445, 0.04060632],
       [0.14542273, 0.6906308 ],
       [0.10200266, 0.63766584]])
np.random.rand(4,2)
array([[0.0215494 , 0.42768197],
       [0.243622  , 0.03437427],
       [0.24093461, 0.43180896],
       [0.81947315, 0.17738195]])
       '''
        self.synaptic_weights=2*np.random.random((3,1))-1

        
        # Describes an s shaped curve we pass the weighted sum of the inputs
    # through this function to normalise them between 0 and 1
    #x is output variable
    def sigmoid(self, x):
        #applying the sigmoid function
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        #computing derivative to the sigmoid function
        #This is the gradient of the Sigmoid curve.
        # It indicates how confident we are about the existing weight.
        return x*(1-x)
    
       #We train the neural network through a process of trial and error.
       # Adjusting the synaptic weights each time.
    def train(self,training_inputs,training_outputs,training_iterations):

        #training the model to make accurate predictions while adjusting
        # Pass the training set through our neural network (a single neuron).
        for iteration in range(training_iterations):
            #siphon the training data via the neuron
            output=self.think(training_inputs)
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error=training_outputs-output

            #performing weight adjustments
            # Multiply the error by the input and again by the gradient of
            # the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes
            # to the weights.
            # Calculate the error for layer 1 (By looking at the weights in
            #layer 1,
            # we can determine by how much layer 1 contributed to the error
            #in layer 2).
            #np.dot([1,2,3],[1,1,1])
            #output:6                                                                                      
            #IF THE INPUT ARRAYS ARE BOTH 1-DIMENSIONAL ARRAYS,
            #NP.DOT COMPUTES THE VECTOR DOT PRODUCT
            # X.T is the transpose of X

            adjustments=np.dot(training_inputs.T,error*self.sigmoid_derivative(output))
            # Adjust the weights.
            self.synaptic_weights+=adjustments
          # The neural network thinks.
    def think(self,inputs):
        #passing the inputs via the neuron to get output
        #converting values to floats

        inputs=inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))

        return output

if __name__=="__main__":

    #initializing the neuron class
     #Intialise a single neuron neural network.
    neural_network=NeuralNetwork()

    print("Beginning randomly generated weights: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 inputs & 1 output
    training_inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_outputs=np.array([[0,1,1,0]]).T

    #training taking place
    # Train the neural network using a training set.
    # Do it 15,000 times and make small adjustments each time.
    neural_network.train(training_inputs,training_outputs,15000)

    print("Ending weights after training: ")
    print(neural_network.synaptic_weights)

    user_input_one=str(input("User Input One: "))
    user_input_two=str(input("User Input Two: "))
    user_input_three=str(input("User Input Three: "))
# Test the neural network with a new situation.
    print("Considering new situation: ",user_input_one,user_input_two,user_input_three)
    print("New output data: ")
    print(neural_network.think(np.array([user_input_one,user_input_two,user_input_three])))

"""
OUTPUT:

Beginning randomly generated weights: 
[[-0.89078318]
 [-0.34733271]
 [-0.49265857]]
Ending weights after training: 
[[10.08710295]
 [-0.20735667]
 [-4.83718482]]
User Input One: 6
User Input Two: 8
User Input Three: 12
Considering new situation:  6 8 12
New output data: 
[0.69371529]
"""

'''
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    
    s = 1/(1 + np.exp(-z))
    
    return s

sigmoid(2)
0.8807970779778823
'''
