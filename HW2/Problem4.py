#!/usr/bin/env python
# coding: utf-8

# # Problem 4

# In[ ]:


# from IPython.display import HTML
# HTML('''<script>
# code_show=true;
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# }
# $( document ).ready(code_toggle);
# </script>
# <a href="javascript:code_toggle()">
# <button>Toggle Code</button></a>''')


# In[ ]:


# import libraries
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# set random seed
np.random.seed(1000)


# In[ ]:


# import mnist data
mnist = sio.loadmat('mnist_all.mat')

# split training and test data
training_data = dict()
test_data = dict()
for key in mnist:
    if "train" in key:
        # store the data under digit key
        training_data[int(key[-1])] = mnist[key]
    if "test" in key:
        # store the data under digit key
        test_data[int(key[-1])] = mnist[key]


# In[ ]:


class NeuralNetwork:
    """
        Create a class that defines a neural network
    """
    def __init__(self,data_input,lr=1e-4,layers=(10,),iterations=2000):
        """
            Initialize the network
        """
        # unpack input into data and labels
        self.data = np.zeros((0,data_input[0].shape[1]))
        self.labels = np.zeros((0,len(data_input)))
        for number in data_input:
            # concatenate data of each digit to array
            self.data = np.concatenate((self.data,data_input[number]),axis=0)
            
            # create labels
            lb = (np.zeros((data_input[number].shape[0],len(data_input))))
            lb[:,number] = 1
            
            # concatenate labels to array
            self.labels = np.concatenate((self.labels,lb),axis=0)
        
        # scale data from [0,255] to [0,1]
        self.data = self.data/255
        
        # add bias term to data_input
        bias = np.ones((self.data.shape[0],1))
        self.data = np.concatenate((bias,self.data),axis=1)
        
        # Save learning rate and iterations
        self.lr = lr
        self.iterations = iterations
        
        # Create list to hold weights in each layer
        self.weights = list()
        
        # Initialize weights in each layer
        for n,layer in enumerate(layers):
            if n == 0:
                # set appropriate dimensions for input layer 
                self.weights.append(np.random.randn(self.data.shape[1],layer)*0.01+0.05)
            else:
                # set appropriate dimensions for each hidden layer
                self.weights.append(np.random.randn(layers[n-1],layer)*0.01+0.05)
        
        # Add final output layer to weights
        self.weights.append(np.random.randn(layers[-1],self.labels.shape[1])*0.01+0.05)
        
        # Display the network dimensions
        print("Input: {}".format(self.data.shape))
        for n,w in enumerate(self.weights):
            print("Layer{}: {}".format(n,w.shape))
        
        # Save MSE
        self.MSE = list()
        
    def _feedforward(self):
        """
            Do feedforward pass on network
        """
        # save output of each layer into a list
        output = list()
        
        # loop over each layer and apply forward pass
        # save output from each layer
        for n,w in enumerate(self.weights):
            if n == 0: # set input to data
                output.append(self._logistic(np.dot(self.data,w)))
            else: # just use last output
                output.append(self._logistic(np.dot(output[n-1],w)))
        
        # return the output
        return output
    
    def _backpropagation(self,output):
        """
            Do backpropagation to calculate gradients
        """
        # calculate error between final output and labels
        error = self.labels - output[1]
        
        # store MSE
        self.MSE.append(0.5*(np.linalg.norm(error,axis=0,ord=2)**2))
        
        # calculate gradients of each layer
        gradients = list()
        gradients.append(error*self._dlog(output[1])) # add final layer gradient
        
        # calculate gradients of other layers
        for n in reversed(range(len(self.weights)-1)):
            # grab gradient from forward layer, dot with weights,
            # and multiply into current gradient
            gradients.append(self._dlog(output[n])*np.dot(gradients[-1],self.weights[n+1].T))
    
        # reverse gradient list
        gradients = gradients[::-1]
        
        # record the sum of gradients (so we can see how well we are converging)
        sum_gradients = [np.linalg.norm(np.linalg.norm(g,axis=0)) for g in gradients]
        print("L2 Gradients: {}".format(sum_gradients))
        
        # update new weights
        for i,grad in enumerate(gradients):
            if i == 0:
                self.weights[i] += self.lr*np.dot(self.data.T, grad)
            else:
                self.weights[i] += self.lr*np.dot(output[i-1].T, grad)
        
    @staticmethod
    def _logistic(data):
        """Logistic Function"""
        return 1/(1 + np.exp(-data))
    
    @staticmethod
    def _dlog(output):
        """Derivative of Logistic Function"""
        return output*(1-output)
    
    def train(self):
        """
            Train the neural network
        """
        # Loop over iterations
        for i in range(self.iterations):
            print("Iteration: {}".format(i))
            # do forward pass
            output = self._feedforward()
        
            # do backpropagation and update gradients
            self._backpropagation(output)
        
            # print latest MSE
            print("MSE: {}".format(self.MSE[-1]))
            print("")
        
        # return output and classes
        return output[-1], np.argmax(output[-1],axis=1)


# In[ ]:


"""
    Train Neural Network
"""
# Create Neural Network
nn = NeuralNetwork(training_data)
out,lbl = nn.train()


# In[ ]:


# Make Figure
plt.figure(figsize=(16,8))
plt.plot(np.array(nn.MSE)[:,0], label='0')
plt.plot(np.array(nn.MSE)[:,1], label='1')
plt.legend()
plt.show()
