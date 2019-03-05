#!/usr/bin/env python
# coding: utf-8

# # Problem 1

# ###### Method
# 
# We use a perceptron to separate out two sets of data: one linearly separable, and the other non-linearly separable. Linearly separable data was created by randomly initializing weights for 50 samples, and then doing class assignments along a predefined threshold boundary. Similarly, the non-linearily separable data was contstructed with 50 samples, with randomly initialized weights. However, each feature was squared before class assignement in order to make the data unseparable in the original space.
# 
# We run the perceptron learning algorithm for both datasets for 1000 iterations or until the error is 0. The data and decision boundaries were plotted, along with the classification error over the number of iterations.

# In[1]:


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<a href="javascript:code_toggle()">
<button>Toggle Code</button></a>''')


# In[2]:


# import libraries
import numpy as np
import matplotlib.pyplot as plt

# set random seed
np.random.seed(1000)


# In[3]:


"""
    Create dataset class
"""
class Dataset:
    """
        Class that creates linear/non-linear datasets
    """
    @classmethod
    def linear(cls, features=2, samples=50):
        """
            Creates linearly separable data
                features: number of features for data
                samples: number of samples
        """
        # create random weight vector based on number of features
        weights = cls._uniform_rand(features)
        
        # now create a bunch of random examples
        examples = cls._uniform_rand(samples,features)
        
        # now do the inner product of the weights with the examples
        decision = np.dot(examples,weights)
        
        # choose a random threshold
        threshold = (decision.max() + decision.min())/2
        class_assignment = 2*((decision > threshold)-0.5)
        
        # return assignment and examples
        return class_assignment,examples
        
    @classmethod
    def nonlinear(cls, features=2, samples=50):
        """
            Creates non-linearly separable data
                features: number of features for data
                samples: number of samples
        """
        # create random weight vector based on number of features
        weights = cls._uniform_rand(features)
        
        # now create a bunch of random examples
        examples = cls._uniform_rand(samples,features)
        
        # square all examples
        examples_squared = examples**2
                            
        # now do the inner product of the weights with the examples
        decision = np.dot(examples_squared,weights)
        
        # set class threshold that balances dataset
        for threshold in np.linspace(decision.min(),decision.max(),100):
            class_assignment = decision > threshold
            # good threshold when at least half of data is one class
            if np.count_nonzero(class_assignment) <= samples/2:
                class_assignment = 2*((decision > threshold)-0.5)
                break
        
        # return assignment and examples
        return class_assignment,examples
    
    @staticmethod
    def _uniform_rand(*args,**kwargs):
        """
            Generates a uniform random variable from -1 to 1
        """
        return (np.random.rand(*args,**kwargs)-0.5)*2
        


# In[4]:


"""
    Create Batch Perceptron
"""
class BatchPerceptron:
    """
        Class implementing a batch perceptron
    """
    def __init__(self, data_input, labels, lr=1, criterion=0.01, iterations=1000):
        """
            Initialize values
        """
        # copy data
        data = np.copy(data_input)
        
        # We add a bias term to the data_input
        bias = np.ones((data.shape[0],1))
        data = np.concatenate((bias,data),axis=1)
        
        # We flip the sign of the negative examples
        data[labels==-1] = -data[labels==-1]
        
        # Assign initialized parameters to object
        parameters = locals()
        for key in parameters:
            if not key == 'self':
                setattr(self, key, parameters[key])
        
        # Initialize weight vector with same dims as inputs
        self.w = np.zeros(data.shape[1])
        
        # Save misclassified
        self.misclassified = list()
        
    def train(self):
        """
            Train Perceptron
        """
        
        # loop through "iterations" or until hit criterion
        for k in range(self.iterations):        
            # get misclassified
            misclassified = np.dot(self.data,self.w) <= 0
            self.misclassified.append(np.count_nonzero(misclassified))
            
            # get gradient
            gradient = np.sum(self.data[misclassified,:],axis=0)
            
            # get the sum of misclassified examples
            status = np.absolute(self.lr*gradient) < self.criterion
            
            # quit if all true
            if np.all(status):
                break
                
            # apply gradient to weights
            self.w = self.w + self.lr*gradient
        
        # output k iteration stats
        print("Finished @ Iteration {}, Weights: {}, Error: {}".format(k,self.w,self.misclassified[-1]/self.data.shape[0]))
        
        # format error
        error = np.array([m/self.data.shape[0] for m in self.misclassified])
        
        # return weights and error
        return self.w, error


# In[5]:


# get data
labels,data = Dataset.linear()
nl_labels,nl_data = Dataset.nonlinear()

# Train perceptron
print('Linearly Separable Data:')
wlin,linerror = BatchPerceptron(data,labels).train()
print('Non-linearly Separable Data:')
wnlin,nlinerror = BatchPerceptron(nl_data,nl_labels).train()

# make x1 values
x1 = np.linspace(-1,1,100)

# plot linear data
x2 = -(wlin[0] + wlin[1]*x1)/wlin[2]
plt.figure(figsize=(16,8))
plt.scatter(data[labels==1][:,0],data[labels==1][:,1],marker='.')
plt.scatter(data[labels==-1][:,0],data[labels==-1][:,1],marker='x')
plt.plot(x1,x2,'g')
plt.title('Linearly Separable Data')
plt.xlim(-1,1); plt.ylim(-1,1)
plt.xlabel('x1'); plt.ylabel('x2')

# plot classification error for linear
plt.figure(figsize=(16,8))
plt.plot(linerror)
plt.title('Classification Error (Linear)'); plt.ylim(0,1)
plt.xlabel('kth Iteration'); plt.ylabel('% Error')

# plot nonlinear data
nl_x2 = -(wnlin[0] + wnlin[1]*x1)/wnlin[2]
plt.figure(figsize=(16,8))
plt.scatter(nl_data[nl_labels==1][:,0],nl_data[nl_labels==1][:,1],marker='.')
plt.scatter(nl_data[nl_labels==-1][:,0],nl_data[nl_labels==-1][:,1],marker='x')
plt.plot(x1,nl_x2,'g')
plt.title('Non-Linearly Separable Data')
plt.xlim(-1,1); plt.ylim(-1,1)
plt.xlabel('x1'); plt.ylabel('x2')

# plot classification error for non-linear
plt.figure(figsize=(16,8))
plt.plot(nlinerror);
plt.title('Classification Error (Non-Linear)'); plt.ylim(0,1)
plt.xlabel('kth Iteration'); plt.ylabel('% Error')

# show plots
plt.show()


# ###### Discussion
# 
# The perceptron algorithm cleanly separates the linearly seperable data in 3 iterations with 0 error. This is to be expected since perceptrons are linear models.
# 
# For non-linearly separable data, the perceptron fails and does not converge to a solution. Classification error for the non-linear data hovers around 50%, which is no better than random chance.
