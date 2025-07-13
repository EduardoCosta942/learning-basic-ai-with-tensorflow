# learning-basic-ai-with-tensorflow
I’m on vacation, but that doesn’t mean I’m going to stop learning. AI is everywhere today, so why not dive into it? As a 10th grader, this journey will be challenging and will require learning college-level math, but with determination, I’m sure I can do it.On this file, you will see my notebook, where the exatly notes I toook while studing are.

# Basics of AI - Neurons (General Idea)
-> On the very base foundation of Artificial Intelligence, we have the neurons. The physiognomy of it is split on 2 parts, the linear transformer and the activation function. Every neuron receives at least 1 parameter (x) which is the topic your neuron will analyze, this can be for example, the size of someone. It is important to notice that the neuron works mainly by applying weights (how much does an input affect the output) and bias (for standard, it will start as 0). 

On the folder "pictures" the archive "basics_of_ai_Neurons_general_idea" is representing the described content.

# Basics of AI - Neurons (Linear Transformer)
-> The linear transformer of a neuron is responsible for applying the weights (w) to the inputs (x), and adding the bias. The formula is: y=∑wᵢxᵢ+b. The name of this module is linear transformer because it will create the best line to fit the data. The upper Greek letter, sigma, is a simpler way to write summations, for example, 1+2+3+4+5+6 can be simplified for ∑xᵢ. The small letter ᵢ indicates the index of the value. 

On the folder "pictures" the archive "basics_of_ai_Neurons_Linear_transformer" is representing the described content.

# Basics of AI - Neurons (Introction for activations functions)
-> The need for an activation function comes from complex datasets, those which can not be described with 1 line, otherwise the prediction will be generalized and low accuracy. In order to solve that problem, we have the activations functions. They were designed for creating curves (functions), so it will fit better the dataset, allowing the AI to learn even more complex patterns. There are numerous activations functions, here there are some of the most used: Sigmoid, ReLU, Leaky ReLU, ELU, Softmax and regression (not an activation function).

On the folder "pictures" the archive "basics_of_ai_Neurons_introduction_for_activations_functions" is representing the described content.

# Basics of AI - Neurons (Activations Functions - Sigmoid)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> The most used activation function for binary classification is the Sigmoid function. This method, represented by the formula "σ(x)=1/(1+e^x)" is capable of transforming the input into values close to 1 or 0, being 1 true and 0 false. More about the mathematics, the constant 'e' is Euler's number and x is the value of 'y' obtained by the linear transformer. Being an output layer activation function, it is not recommended to use it on hidden layers to avoid the Vanishing Gradient, an error caused by values close to 0 while doing the backpropagation, which can block the AI model to learn.

On the folder "pictures" the archive "basics_of_ai_Neurons_sigmoid" is representing the described content.

# Basics of AI - Neurons (Activations Functions - ReLU, Leaky ReLU and ELU)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> Most used function for hidden layers since it is computationally efficient, justified by it considering inputs values less than 0 as 0. The formula of this function is "f(x) = max(0,x)", which means it is converting every value under than 0 on the x-axis for a zero y-axis. Since this function zeroes some values, it is non-linear function, which means that by summing 2 ReLUs, it may create a curve, fitting better the data. This activation function also has a problem, the Dying ReLu. The neuron network may stop learning if there are many inputs values under than 0, what can approximate the output to 0 and harm the learning process. If this gets to happen, use Leaky ReLU "f(x) = max(0.1x, x)" or ELU "(x if x>0 else σ(e^x -1), x)" (both of them avoid this issue but need more computational power)

On the folder "pictures" the archive "basics_of_ai_Neurons_full_relu" is representing the described content.

# Basics of AI - Neurons (Softmax)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> This function is mostly used on the output layer when it is nescessary to classify something into a category. It splits the data into a vector and convert the values into a probabilities of belonging to each group, the sum of the probabilities will always be 1. Formula: σ(zᵢ)=e^(zᵢ)/∑e^(zⱼ)

On the folder "pictures" the archive "basics_of_ai_Neurons_softmax" is representing the described content.

# Basics of AI - Neurons (Regression - not a function)
-> When we use values that do not require binary classification or grouping, we are dealing with real values. In this case, no activation function is needed in the output layer, simply summing the inputs and adding the bias is enough. This is because applying an activation function in the output layer could produce a non-linear result, which is beneficial for accuracy when needed, but not necessary for real-valued outputs.

# Neural Network - Introduction
-> Depending on the output you want, it may require a neural network. Inspired by the structure of the human brain, this scheme connects different neurons in layers (Input layer, Hidden Layers, Output Layers), being capable of learning patterns and relationships hidden in the data. By combining many neurons, this approach becomes a foundation of modern AI applitactions that recognize images, understand speech and make predictions without a linear relations.

On the folder "pictures" the archive "neural_network_introduction" is representing the described content.

# Neural Network - Foundations of neural network (Input layer)
-> The input layer is composed of at least 1 input the AI will analyze, just like the input of a sinle neuron, stated at the start of this page. The difference is that in a neural network, there will be more than one neuron to analyze the same input, increasing the capacity of the AI to find more complex patterns and to learn. The input should be distributed to all neurons, as shown by the picture.

On the folder "pictures" the archive "neural_network_input_layer" is representing the described content.

# Neural network - Foundations of neural network (Hidden layers)
-> The layer responsable for processing the data and applying the activation function in order to find complex patterns represented by non-linear relationships. It suitable noticing that every neuron from the same layer might receive the same input, however, they do have differents weights and biases, again, allowing the network to find specific patterns. It is important to mention that there can be more than 1 hidden layer, and the layer t will be used as parameter for layer t+1, following the same logic of the input layer.

On the folder "pictures" the archive "neural_network_hidden_layer" is representing the described content.

# Neural network - Foundations of neural network (Output layer)
-> The output layer is the last layer of the neural network, here a neuron is going to apply weights and bias for the inputs fornicated by the last hidden layer, getting together the calculations of every neuron. The amount of neurons in this layer depends on how many inputs are needed (1 output, 1 neuron). As the last layer, this has the responsability of transforming the data for the context, this can be done by using the activations functions that are specific for output layers.

# Neural network - Feedfoward
-> The feedfoward is the process of getting the input and make every layer proecess this data, always using the previous layer (except for the input layer) as input. This process is the most commum, seen that if there is an output, it must have passed thourgh every layer.

# Neural network - Backpropagation (introduction)
-> Everything we have seen until now tells us how the output is processed, but, how does an AI model learn in fact? As stated previously, every neuron applies differents weights and biases to the data. Those values usually start with a random value, the neural network will just learn something when there is an update of them, making the result of each layer even more precise and reliable.

# Neural network - Backpropagation (Mathematic foundations - Chain Rule - Derivative)
-> One of the most used mathematic foundations for appying the backpropagation is the Chain Rule, but before that, what is a derivative? A derivative tells us the slope of the tangent to a curve (On AIs, the curve will be the non-linear representation after the activation function). The formula of it is: f'(x)=(f(x+h)-f(x))/h as h → 0. The reason why we use the derivative is because it measures how the function changes according to the input, in other words, the slope of the curve at each point.

# Neural network - Backpropagation (Mathematic foundations - Chain Rule)
-> The Chain Rule is a procedure to connect a function to another by a similar parameter. By this, it is possible to calculate the derivative of a funcion composed by another. For AI models, this help by allowing to identify how the information has been changed comparing the previous layers and neurons. The formula of this procedure is: f(g(x))=f'(gx)*g(x). An important thing to mention is that whenever we are using derivatives, we should represent them with the prefix: dy/dx, where y and x may vary according to the variale.

# Neural network - Backpropagation (Mathematic foundations - Gradient Descent)
-> The Gradient Descent is an optimization algorithm used to reduce the error between the predicted output and the real dataset. It works by calculating the gradient of the loss function with respect to the parameters and updating them until the error level is as close to 0 as possible.

There are many ways to define the loss function, but the one I’ve learned is the Sum of Squared Residuals (SSR). Basically, the SSR is the sum of the squares of the differences between each actual data point and the predicted value: SSR = Σ(yᵢ − ŷᵢ)²

By graphing the SSR (x = parameter, y = SSR) as the parameters change, the plot usually has a bowl shape. The point where the derivative of the loss function with respect to the parameter is closest to 0 represents the minimum error found by gradient descent.
# Neural Network - Backpropagation
-> The backpropagation is named like this because we start updating the values (weights and biases) from the output layer until the first layer of the hidden layer. The updated value is: θₜ₊₁=θₜ-η(d SSR / d θ). This means that the parameter is going to decrease the learning rate times the derivative of the loss function with respect to the parameter. This will only work since the gradient indicates how far the parameter is from the loss function and mutiplying it by the learning rate allows the parameter to be updated slowly, increasing the precision of the gradient.

How far the current layer is from the output layer also indicates the nescessity of derivating in a different way. The objective of the Chain Rule is to connect a data to another data using a comumn parameter, this also have to be done for neural networks. By a simple look, it is easy to notice that every neuron is connected somehow, just as the guide picture. To conect the loss function to the w1 of the example, we might run every intermediate layer: d e / d w1 = (d e / d f2) * (d f2 / d z2) * (d z2 / d f1) * (d f1 / d z1) * (d z1 / d w1), observe that it just connects to the activations functions, summatories and the specific wheigh/bias you will update. 

# Credits
Youtube Channel: Statquest with Josh Starmer
Youtube Channel: codebasics
Artificial Inteligence Assistant: ChatGPT - OpenAi

# Observation
Content developed on 06/07/2025 to 11/07/2025
The next step of my learning is available on the .md file at the directory "practing".