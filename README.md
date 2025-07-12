# learning-basic-ai-with-tensorflow
I’m on vacation, but that doesn’t mean I’m going to stop learning. AI is everywhere today, so why not dive into it? As a 10th grader, this journey will be challenging and will require learning college-level math, but with determination, I’m sure I can do it.On this file, you will see my notebook, where the exatly notes I toook while studing are.

# Basics of AI - Neurons (General Idea)
-> On the very base foundation of Artificial Inteligence, we have the neurons. The physiognomy of it is splited on 2 parts, the linear transformer and the activation funcion. Every neuron receives at least 1 parameter (x) which is the topic your neuron will analyze, this can be for example, the size of someone. It is important to notice that the neuron works mainly by applying weights (how much does an input affect the output) and bias (for standard, it will start as 0). 

On the folder "pictures" the archive "basics_of_ai_Neurons_general_idea" is representing the described content.

# Basics of AI - Neurons (Linear Transformer)
-> The linear transformer of a neuron is responsible for applying the weights (w) for the inputs (x), summing with the bias. The formula of it is: y=∑wᵢxᵢ+b. The name of this module is linear transformer because it will create the best line to fit the data. The upper Greek letter, sigma, is a simpler way to write summations, for example, 1+2+3+4+5+6 can be simplified for ∑xᵢ. The small letter ᵢ indicates the index of the value. 

On the folder "pictures" the archive "basics_of_ai_Neurons_Linear_transformer" is representing the described content.

# Basics of AI - Neurons (Introction for activations functions)
-> The nescessity for a activation function comes from complex datasets, those which can not be described with 1 line, otherwise the prediction will be generalized and low accuracity. In order to resolve that problem, we have the activations functions. They were designed for creating lines with curves (not a line), so it will fit better the dataset, allowing the AI to learn even more complex patterns. There are imnumerous activations functions, here there are some of the most used: Sigmoid, ReLU, Leaky ReLU, ELU, Softmax and regression (not a activation function).

On the folder "pictures" the archive "basics_of_ai_Neurons_introduction_for_activations_functions" is representing the described content.

# Basics of AI - Neurons (Activations Functions - Sigmoid)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> Most used activation function for binary classification. This method, represented by the formula "σ(x)=1/(1+e^x)" is capable of transforming the input into 1 or 0, being 1 true and 0 false. More about the mathematics, the const 'e' is the Euler's number and x is the value of 'y' obtained by the linear transformer. Being an output layer activation function, it is not recommended to use it on hidden layers to avoid the Vanishing Gradient, an error caused by values close to 0 while doing the backpropagation, what can block the AI model to learn.

On the folder "pictures" the archive "basics_of_ai_Neurons_sigmoid" is representing the described content.

# Basics of AI - Neurons (Activations Functions - ReLU, Leaky ReLU and ELU)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> Most used function for hidden layers since it is computely efficient, justified by it considering inputs under than 0 as 0. The formula of this function is "f(x) = max(0,x)", which means it is converting every value on the 'x' axis for a zero 'y'. Since this function zeroes some values, it is not a linear function, which means that by summing 2 ReLUs, it may deal to a curve, fitting better the data. This activation function also has a problem, the Dying ReLu. The neuron network may stop learning if there are many inputs values under than 0, what can approximate the output to 0 and prejudicate the learning process. If this gets to happen, use Leaky ReLU "f(x) = max(0.1x, x)" or ELU "(x if x>0 else σ(e^x -1), x)" (both of them avoid this issue but need more computational power)

On the folder "pictures" the archive "basics_of_ai_Neurons_full_relu" is representing the described content.

# Basics of AI - Neurons (Softmax)
RECOMENDATION: Skip this content and try this after seeing the backpropagation
-> This function is mostly used on the output layer when there is the nescessity to classify something into a category. It split the data into a vector and convert the values into a probability of belonging to each group, the sum of the probabilities will always be 1. Formula: σ(zᵢ)=e^(zᵢ/∑e^zⱼ)

On the folder "pictures" the archive "basics_of_ai_Neurons_softmax" is representing the described content.

# Basics of AI - Neurons (Regression - not a function)
-> Whenever we are using values that do not need a binary classification or a group division, we are talking about real values and they do not need an activation function, because it will be only for the output layer, where only summing the inputs and adding the bias is enough. It suitable seeing that if there is any activation function, the output layer can deal to a non-linear result, which is good due to its accuracy.