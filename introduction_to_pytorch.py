## NOTE: Even though we use the PyTorch module, we import it with the name 'torch', which was the original name.
import torch # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.

import matplotlib.pyplot as plt ## matplotlib allows us to draw graphs.
import seaborn as sns ## seaborn makes it easier to draw nice-looking graphs.

# # Build a Simple Neural Network in PyTorch
# 
# Building a neural network in **PyTorch** means creating a new class with two methods: `__init__()` and `forward()`. The `__init__()` method defines and initializes all of the parameters that we want to use, and the `forward()` method tells **PyTorch** what should happen during a forward pass through the neural network.

## create a neural network class by creating a class that inherits from nn.Module.
class BasicNN(nn.Module):

    def __init__(self): # __init__() is the class constructor function, and we use it to initialize the weights and biases.
        
        super().__init__() # initialize an instance of the parent class, nn.Model.
        
        ## Now create the weights and biases that we need for our neural network.
        ## Each weight or bias is an nn.Parameter, which gives us the option to optimize the parameter by setting
        ## requires_grad, which is short for "requires gradient", to True. Since we don't need to optimize any of these
        ## parameters now, we set requires_grad=False.
        ##
        ## NOTE: Because our neural network is already fit to the data, we will input specific values
        ## for each weight and bias. In contrast, if we had not already fit the neural network to the data,
        ## we might start with a random initalization of the weights and biases.
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        
        
    def forward(self, input): ## forward() takes an input value and runs it though the neural network 
                              ## illustrated at the top of this notebook. 
        
        ## the next three lines implement the top of the neural network (using the top node in the hidden layer).
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        
        ## the next three lines implement the bottom of the neural network (using the bottom node in the hidden layer).
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        
        ## here, we combine both the top and bottom nodes from the hidden layer with the final bias.
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        output = F.relu(input_to_final_relu)
    
        return output # output is the predicted effectiveness for a drug dose.


# Once we have created the class that defines the neural network, we can create an actual neural network and print out its parameters, just to make sure things are what we expect.


## create the neural network. 
model = BasicNN()

## print out the name and value for each parameter
for name, param in model.named_parameters():
    print(name, param.data)


# ## BAM!!!
# The values for each weight and bias in `BasicNN` match the values we see in the optimized neural network (below).
# # Use the Neural Network and Graph the Output
# 
# Now that we have a neural network, we can use it on a variety of doses to determine which will be effective. Then we can make a graph of these data, and this graph should match the green bent shape fit to the training data that's shown at the top of this document. So, let's start by making a sequence of input doses...

## now create the different doses we want to run through the neural network.
## torch.linspace() creates the sequence of numbers between, and including, 0 and 1.
input_doses = torch.linspace(start=0, end=1, steps=11)

# now print out the doses to make sure they are what we expect...
input_doses


# Now that we have `input_doses`, let's run them through the neural network and graph the output...

## create the neural network. 
model = BasicNN() 

## now run the different doses through the neural network.
output_values = model(input_doses)

## Now draw a graph that shows the effectiveness for each dose.
##
## First, set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses, 
             y=output_values, 
             color='green', 
             linewidth=2.5)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

## optionally, save the graph as a PDF.
# plt.savefig('BasicNN.pdf')


# The graph shows that the neural network fits the training data. In other words, so far, we don't have any bugs in our code.
# # Double BAM!!!

# # Optimize (Train) a Parameter in the Neural Network and Graph the Output
# 
# Now that we know how to create and use a simple neural network, and we can graph the output relative to the input, let's see how to train a neural network. The first thing we need to do is tell **PyTorch** which parameter (or parameters) we want to train, and we do that by setting `requires_grad=True`. In this example, we'll train `final_bias`.

## create a neural network by creating a class that inherits from nn.Module.
## NOTE: This code is the same as before, except we changed the class name to BasicNN_train and we modified 
##       final_bias in two ways:
##       1) we set the value of the tensor to 0, and
##       2) we set "requires_grad=True".
class BasicNN_train(nn.Module):

    def __init__(self): # __init__ is the class constructor function, and we use it to initialize the weights and biases.
        
        super().__init__() # initialize an instance of the parent class, nn.Module.
        
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        ## we want to modify final_bias to demonstrate how to optimize it with backpropagation.
        ## The optimal value for final_bias is -16...
#         self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
        ## ...so we set it to 0 and tell Pytorch that it now needs to calculate the gradient for this parameter.
        self.final_bias = nn.Parameter(torch.tensor(0.), requires_grad=True) 
        
    def forward(self, input):
        
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
    
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        output = F.relu(input_to_final_relu)
        
        return output


# Now let's graph the output of `BasicNN_train`, which is currently not optimized, and compare it to the graph we drew earlier of the optimized neural network.


## create the neural network. 
model = BasicNN_train() 

## now run the different doses through the neural network.
output_values = model(input_doses)

## Now draw a graph that shows the effectiveness for each dose.
##
## set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses, 
             y=output_values.detach(), ## NOTE: because final_bias has a gradident, we call detach() 
                                       ## to return a new tensor that only has the value and not the gradient.
             color='green', 
             linewidth=2.5)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

## lastly, save the graph as a PDF.
# plt.savefig('BasicNN_train.pdf')

## create the training data for the neural network.
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])


# ...and now let's use that training data to train (or optimize) `final_bias`.

## create the neural network we want to train.
model = BasicNN_train()

optimizer = SGD(model.parameters(), lr=0.1) ## here we're creating an optimizer to train the neural network.
                                            ## NOTE: There are a bunch of different ways to optimize a neural network.
                                            ## In this example, we'll use Stochastic Gradient Descent (SGD). However,
                                            ## another popular algortihm is Adam (which will be covered in a StatQuest).

print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")

## this is the optimization loop. Each time the optimizer sees all of the training data is called an "epoch".
for epoch in range(100):
        
    ## we create and initialize total_loss for each epoch so that we can evaluate how well model fits the
    ## training data. At first, when the model doesn't fit the training data very well, total_loss
    ## will be large. However, as gradient descent improves the fit, total_loss will get smaller and smaller.
    ## If total_loss gets really small, we can decide that the model fits the data well enough and stop
    ## optimizing the fit. Otherwise, we can just keep optimizing until we reach the maximum number of epochs. 
    total_loss = 0
    
    ## this internal loop is where the optimizer sees all of the training data and where we 
    ## calculate the total_loss for all of the training data.
    for iteration in range(len(inputs)):
        
        input_i = inputs[iteration] ## extract a single input value (a single dose)...
        label_i = labels[iteration] ## ...and its corresponding label (the effectiveness for the dose).
        
        output_i = model(input_i) ## calculate the neural network output for the input (the single dose).
        
        loss = (output_i - label_i)**2 ## calculate the loss for the single value.
                                       ## NOTE: Because output_i = model(input_i), "loss" has a connection to "model"
                                       ## and the derivative (calculated in the next step) is kept and accumulated
                                       ## in "model".
        
        loss.backward() # backward() calculates the derivative for that single value and adds it to the previous one.
        
        total_loss += float(loss) # accumulate the total loss for this epoch.
        
        
    if (total_loss < 0.0001):
        print("Num steps: " + str(epoch))
        break
      
    optimizer.step() ## take a step toward the optimal value.
    optimizer.zero_grad() ## This zeroes out the gradient stored in "model". 
                          ## Remember, by default, gradients are added to the previous step (the gradients are accumulated),
                          ## and we took advantage of this process to calculate the derivative one data point at a time.
                          ## NOTE: "optimizer" has access to "model" because of how it was created with the call 
                          ## (made earlier): optimizer = SGD(model.parameters(), lr=0.1).
                          ## ALSO NOTE: Alternatively, we can zero out the gradient with model.zero_grad().
    
    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")
    ## now go back to the start of the loop and go through another epoch.

print("Total loss: " + str(total_loss))
print("Final bias, after optimization: " + str(model.final_bias.data))


# So, if everything worked correctly, the optimizer should have converged on `final_bias = 16.0019` after **34** steps, or epochs. **BAM!**
# 
# Lastly, let's graph the output from the optimized neural network and see if it's the same as what we started with. If so, then the optimization worked.

## run the different doses through the neural network
output_values = model(input_doses)

## set the style for seaborn so that the graph looks cool.
sns.set(style="whitegrid")

## create the graph (you might not see it at this point, but you will after we save it as a PDF).
sns.lineplot(x=input_doses, 
             y=output_values.detach(), ## NOTE: we call detach() because final_bias has a gradient
             color='green', 
             linewidth=2.5)

## now label the y- and x-axes.
plt.ylabel('Effectiveness')
plt.xlabel('Dose')

## lastly, save the graph as a PDF.
# plt.savefig('BascNN_optimized.pdf')


# And we see that the optimized model results in the same graph that we started with, so the optimization worked as expected.

# # Triple BAM!!!
