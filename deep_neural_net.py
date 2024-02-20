import timeit
import numpy as np
from typing import List
from scipy.special import expit


# Datasets
# http://yann.lecun.com/exdb/mnist/ -- mnist dataset hardLatest to work with but i'll figure it out! TODO
# https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download -- csv form of mnist data easy to work with!

WRITE = True

def write(*args, **kwargs):
    if (WRITE):
        print(*args, **kwargs)

# Neural Network Class Definition
class NeuralNetwork:
    # initialize neural network like setting input layer, hidden layers & output layer.
        # set number of nodes in each input, hidden & output layer
    def __init__(self, 
                inputnodes:int=None, 
                hiddennodes:List[int]=None, 
                outputnodes:int=None, 
                learningrates:float=None):
        self.inodes = inputnodes
        self.hnodes = hiddennodes # list of hidden nodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrates

        # layer output capture
        self.hidden_outputs = {}

        # layer error capture
        self.hidden_errors = {}

        # now we need to create link weights matrix from one layer to next
        # ex- w_input_hidden matrix will have no. of hidden layer neurons times rows and no. of input layer neurons times col.
        # (not the best way or general way) link weights can have value from -0.5 to +0.5, instead 0 to 1.
        # self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        # self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        ## setting random weights 
        """
        from viewpoint of input layers
            i1 [h1, h2, h3, ..., hn]
            i2 [h1, h2, h3, ..., hn]
            i3
            ...
            im
        
        from viewpoint of hidden layer

            h1 [ Wi1, Wi2, Wi3, ..., Win ]
            h2 [ Wi1, Wi2, Wi3, ..., Win ]
            ...
            hm 
        
        """
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes[0], self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes[-1], -0.5), (self.onodes, self.hnodes[-1]))
        self.whh = {
            f"wh{i}h{i+1}" : np.random.normal(0.0, pow(self.hnodes[i-1], -0.5), (self.hnodes[i], self.hnodes[i-1]))  
            for i in range(1, len(self.hnodes)) }

        # activation function is the sigmoid function
        self.activation_function = lambda x: expit(x)

        write("\n\nself.wih : ", self.wih)
        write("self.wih.shape: ", self.wih.shape)

        write("\n\nself.who : ", self.who)
        write("self.who.shape: ", self.who.shape)

        write("\n\nself.whh : ", self.whh)
        write("self.whh['wh1h2'].shape: ", self.whh.get('wh1h2', self.who).shape)


    # after setting up a neural network, train the neural network
    def train(self, input_list, target_list):
        # Theory
        # dE/dWij = -2(t_k-o_k)*sigmoid(summation(Wij.oj))*(1-sigmoid(summation(Wij.oj)))*o_j
        # t_k-o_k = e_k | Error calculation
        # new_weight = old_weight - self.lr*(dE/dWij)
        # new_weight = old_weight + self.lr*( (t_k-o_k)*sigmoid(summation(Wij.oj) )*( 1-sigmoid(summation(Wij.oj)))*o_j )
        # final_outputs = sigmoid(summation(Wij.oj))), output_error = (t_k-o_k)
        # self.who = self.who + self.lr*np.dot((output_error*final_outputs*(1.0-final_outputs)), np.transpose(self.hidden_outputs[f'h{len(self.hnodes)}']))

        # converted inputs_list to 2d array, basically converted/transposed a 1 by n array to n by 1 array
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        write("\n\ninputs: ", inputs)
        write("targets: ", targets)



        hidden_layer_1_inputs = np.dot(self.wih, inputs)
        self.hidden_outputs['h1'] = self.activation_function(hidden_layer_1_inputs)

        write("\n\nhidden_layer_1_inputs: ", hidden_layer_1_inputs)
        write("self.hidden_outputs['h1']: ", self.hidden_outputs['h1'])

        """
        E.G.

        I --- H1 --- H2 --- H3 --- O


        """

        for i in range(1, len(self.hnodes)):
            layer_number = i+1
            self.hidden_outputs[f'h{layer_number}'] = self.activation_function(
                np.dot(
                    self.whh[f'wh{i}h{i+1}'], # hidden to hidden layer mesh
                    self.hidden_outputs[f'h{i}'] # previous hidden layer outputs
                    )
                )
            write(f"\n\nself.hidden_outputs[f'h{layer_number}']: ", self.hidden_outputs[f'h{layer_number}'])
            write(f"self.hidden_outputs[f'h{layer_number}'].shape: ", self.hidden_outputs[f'h{layer_number}'].shape)
                
        final_inputs = np.dot(self.who, self.hidden_outputs[f'h{len(self.hnodes)}'])
        final_outputs = self.activation_function(final_inputs)
        write("\n\nfinal_inputs: ", final_inputs)
        write("final_outputs: ", final_outputs)

        # Backpropogation
        """
        E.G.

        I --- H1 --- H2 --- H3 --- O


        e(O) = targets - outputs

        e(H3) = Tran( Wei( H3 --- O) ) @ e(O)
        
        corollary, 

        e(H2) = Tran( Wei( H2 --- H3) ) @ e(H3)
        
        e(H1) = Tran( Wei( H1 --- H2) ) @ e(H2)

        """
        output_error = targets-final_outputs

        write("\n\ntargets: ", targets)
        write("output_error: ", output_error)
        # hidden_error = last_hidden_layer_error = np.dot(self.who.T, output_error)
        self.hidden_errors[f'h{len(self.hnodes)}'] = np.dot(self.who.T, output_error)
        write(f"\n\nself.hidden_errors[f'h{len(self.hnodes)}']: ", self.hidden_errors[f'h{len(self.hnodes)}'])
        write(f"self.hidden_errors[f'h{len(self.hnodes)}'].shape: ", self.hidden_errors[f'h{len(self.hnodes)}'].shape)

        self.who += self.lr*np.dot((output_error*final_outputs*(1.0-final_outputs)), np.transpose(self.hidden_outputs[f'h{len(self.hnodes)}']))
        write(f"\n\nself.who: ", self.who)
        write(f"self.who.shape: ", self.who.shape)
        
        for i in range(len(self.hnodes), 1, -1):
            self.hidden_errors[f'h{i-1}'] = np.dot(self.whh[f'wh{i-1}h{i}'].T, self.hidden_errors[f'h{i}'])
            self.whh[f'wh{i-1}h{i}'] += self.lr*np.dot((self.hidden_errors[f'h{i}']*self.hidden_outputs[f'h{i}']*(1.0-self.hidden_outputs[f'h{i}'])), np.transpose(self.hidden_outputs[f'h{i-1}']))
            write(f"\n\nself.hidden_errors[f'h{i-1}']: ", self.hidden_errors[f'h{i-1}'])
            write(f"self.whh[f'wh{i-1}h{i}']: ", self.whh[f'wh{i-1}h{i}'])
            write(f"self.whh[f'wh{i-1}h{i}'].shape: ", self.whh[f'wh{i-1}h{i}'].shape)


        self.wih += self.lr*np.dot((self.hidden_errors[f'h1']*self.hidden_outputs[f'h1']*(1.0-self.hidden_outputs[f'h1'])), np.transpose(inputs))
        write(f"self.wih: ", self.wih)
        write(f"self.wih.shape: ", self.wih.shape)


    # query the neural network
    def query(self, input_list):
        # take inputs & convert inputs into an ndarray object
        inputs = np.array(input_list, ndmin=2).T # u are using ndmin=2 because you want the inputs to be in row instead of col.
        # also, because you are multipling a (m by n) matrix by (x by n), so using ndmin=2 make (x by n) to (n by x)

        # calculate dot product from input layer to hidden layer with activation function applied
        self.hidden_outputs['h1'] = self.activation_function(np.dot(self.wih, inputs))

        """
        E.G.

        I --- H1 --- H2 --- H3 --- O

        """

        for i in range(1, len(self.hnodes)):
            layer_number = i+1
            self.hidden_outputs[f'h{layer_number}'] = self.activation_function(
                np.dot(
                    self.whh[f'wh{i}h{i+1}'], # hidden to hidden layer mesh
                    self.hidden_outputs[f'h{i}'] # previous hidden layer outputs
                    )
                )

        # calculate dot product from hidden layer to output layer with activation function applied                
        final_output = self.activation_function(np.dot(self.who, self.hidden_outputs[f'h{len(self.hnodes)}']))

        formated_final_output = np.array(final_output.T, ndmin=1)

        return final_output, np.argmax(formated_final_output)


def extract_input_data(data_list = []):
    input_data = []
    for data_str in data_list:
        label, *pixel_list = data_str.strip("\n").split(",")
        pixel_list = (np.asfarray(pixel_list)/255) * 0.99 + 0.01 # Here, to avoid saturation conditions we have used 0.99 and 0.01, 
        # e.g. if a 0 value came then 0/255 * 099 = 0 and this created a saturation of values in the model, so we added 0.01
        temp_target_list = np.zeros(10)+0.01
        temp_target_list[int(label)] = 0.99
        input_data.append((pixel_list, temp_target_list))
    return input_data



if __name__ == "__main__":
    WRITE = False
    input_layer_nodes = 28*28 # 784
    hidden_layer_nodes = [64, 32]
    output_layer_nodes = 10
    learning_rate = 0.05
    epochs = 4 # no. of times model should be trained on training data
    n = NeuralNetwork(input_layer_nodes, hidden_layer_nodes, output_layer_nodes, learning_rate)
    # output_layer_result = n.query([1.0, -0.5, -1.5])
    # print("output_layer:\n", output_layer_result)

    with open('mnist_csv/mnist_train.csv', 'r') as training_data_file:
        training_data_list = training_data_file.readlines()[1:]
    
    converted_raw_csv_path = "./csv_converted_images/test_raw.csv"
    mnist_test_set = "./mnist_csv/mnist_test.csv"

    with open(mnist_test_set, 'r') as test_data_file:
        test_data_list = test_data_file.readlines()[1:]

    # print(len(training_data_list))
    # print(training_data_list[0])

    # label, *pixel_grid = training_data_list[3].strip("\n").split(",")
    # # print(label, pixel_grid)
    # image_array = np.asfarray(pixel_grid).reshape(28, 28)
    # print(image_array)

    # plt.imshow(image_array, cmap='Greys', interpolation=None)
    # plt.show()
    
    training_data = extract_input_data(data_list=training_data_list)
    test_data = extract_input_data(data_list=test_data_list)


    # print(f"""

    # Phase 1: checking initial performance
    # Output_value | Target_value
    # """)
    # scorecard = []
    # for input, target in training_data:
    #     out_arr, label = n.query(input)
    #     target_out = np.argmax(target)
    #     print(label, target_out)
    #     if int(label) == int(target_out):
    #         scorecard.append(1)
    #     else:
    #         scorecard.append(0)
    # print(f"performance:= {sum(scorecard)/len(scorecard)}")

    print(f"""

    Phase 2: training the NN

    """)
    scorecard = []
    t1 = timeit.default_timer()
    for epoch in range(epochs):
        print(f"""\t\tepoch {epoch} started""")
        t2 = timeit.default_timer()
        for input, target in training_data:
            n.train(input, target)
        t3 = timeit.default_timer()
        print(f"""\t\tepoch {epoch} training time: {t3-t2} seconds\n""")
    t4 = timeit.default_timer()
    print(f"""

    Phase 2: training ended
    total training time: {t4-t1} seconds

          
    """)

    print(f"""

    Phase 3: checking after training performance (test_performance)
    Output_value | Target_value

    """)
    t5 = timeit.default_timer()
    scorecard = []
    for input, target in test_data:
        out_arr, label = n.query(input)
        target_out = np.argmax(target)
        # print(label, target_out)
        if int(label) == int(target_out):
            scorecard.append(1)
        else:
            scorecard.append(0)
    t6 = timeit.default_timer()
    print(f"performance efficiency:= {sum(scorecard)/len(scorecard)*100}%")
    print(f"\nTotal testing time: {t6-t5} seconds")







# Problem 1:
    # TODO: Save the weights, get the insights, check there learning flow, take there derivative = 0

# Answer Theory:
    # Need to save initial set of link weights values that are random in the sense that they close to one as they 

# Problem 2:
    # TODO: Make it general purpose FFNN with input_nodes, list of hidden_nodes, output_nodes, lr, epochs, training file, test_file
        # TODO: Also, try to save the "hyperparam" file as json object or csv object and if user already has csv/json of 
        # "hyperparam" then take the file as a input, no need to train the data, set provided weights on the network.



