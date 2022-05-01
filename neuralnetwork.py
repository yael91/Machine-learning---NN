import numpy as np

lr = 0.001
pic_size = 784
hidden_size = 50
biggest_value_x = 225.0
num_classes = 10

# the function is shuffling the data
def shuffle_files(file_x, file_y):
    zip_input = list(zip(file_x, file_y))
    np.random.shuffle(zip_input)
    shuffle_x, shuffle_y = zip(*zip_input)
    return shuffle_x, shuffle_y


# Differentiate the ReLU function. if the x value greater than 0 then the derivation is 1, else - 0
def dReLU(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            derivative = 0
            if (x[i][j] > 0):
                derivative = 1
            x[i][j] = derivative
    return x


# in this function we calculate the ReLU function
def reLU(x):
    output = np.maximum(x, np.zeros(np.shape(x)))
    return output


# in this function we calculate the softmax function
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# this function create a vector of x (the input picture)
def set_vector_input(x):
    x = np.array(x)
    x.shape = (pic_size, 1)
    return x

# this function create a vector of y (is vector with 10 cells - one to each class from Fashion-MNIST)
def set_vector_output(y):
    y_output = np.zeros(num_classes)
    y_output[np.int(y)] = 1
    return y_output


# this function calculate the forward propagation algorithm
def fprop(x, y, params, option):
    x = set_vector_input(x)
    # pass each picture from the input layer through one hidden layer and then to the output layer
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    h1 = reLU(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)

    # option 0 -> we do validation or the test algorithm
    if (option == 0):
        return h2
    # option 1 -> we do train algorithm
    else:
        y_output = set_vector_output(y)
        # this is the loss formula to multi class
        loss = -1 * np.dot(y_output, np.log(h2))
        ret = {'x': x, 'y_output': y_output, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
        # add the values of params to ret dictionary
        for key in params:
            ret[key] = params[key]
        return ret


# calculate the backward propagation algorithm
def bprop(fprop_output):
    x, y_output, z1, h1, z2, h2, loss = [fprop_output[key] for key in ('x', 'y_output', 'z1', 'h1', 'z2', 'h2', 'loss')]
    y_output = np.asarray(y_output)
    y_output.shape = (num_classes, 1)

    # calculate all the gradients from the output layer through the hidden layer and then to the input layer
    dz2 = (h2 - y_output)
    dW2 = np.dot(dz2, h1.T)
    db2 = dz2
    dh1 = np.dot(fprop_output['W2'].T, dz2)
    dz1 = dReLU(z1)
    dz1 = np.multiply(dh1, dz1)
    dW1 = np.dot(dz1, x.T)
    db1 = dz1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


# update the 'W' and 'b' matrix
def new_weight(matrix, derivatives):
    matrix = matrix - np.multiply(lr, derivatives)
    return matrix


# calculate the train algorithm
def run_train_files(train_x, train_y, params):
    loss_arr = []
    for x, y in zip(train_x, train_y):
        # norm x
        x = x / biggest_value_x
        # forward
        ret = fprop(x, y, params, 1)
        train_loss = ret['loss']
        # back propagation
        gradients = bprop(ret)
        # update the W's and b's matrix
        W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
        dW1, db1, dW2, db2 = [gradients[key] for key in ('W1', 'b1', 'W2', 'b2')]
        params['W1'] = new_weight(W1, dW1)
        params['b1'] = new_weight(b1, db1)
        params['W2'] = new_weight(W2, dW2)
        params['b2'] = new_weight(b2, db2)
        loss_arr.append(train_loss)
    # find the mean loss
    mean_loss = np.mean(loss_arr)
    return [params, mean_loss]


# this function calculate the validation algorithm
def run_validation_files(validation_x, validation_y, params):
    correct_counter = 0.0
    total_num_pic = 0.0
    for x, y in zip(validation_x, validation_y):
        # norm x
        x = x / biggest_value_x
        # calculate the forward algorithm till the level of softmax function
        prediction = fprop(x, y, params, 0)
        # add 1 to the counter of correct_counter if the prediction is equals to y, else add 1 to error_counter
        if prediction.argmax() == y:
            correct_counter = correct_counter + 1
        total_num_pic += 1
    # calculates the percentage of times the algorithm output was correct
    correct_average = correct_counter / total_num_pic
    return correct_average


# the main function
def main():
    # load files
    train_x = np.loadtxt("train_x_short")
    train_y = np.loadtxt("train_y_short")
    #test_x = np.loadtxt("test_x")

    # shuffle files
    train_x, train_y = shuffle_files(train_x, train_y)
    # create validation and test files
    size_validation = int(0.2 * np.size(train_y))
    validation_x = train_x[-size_validation:]
    validation_y = train_y[-size_validation:]
    train_x = train_x[:-size_validation]
    train_y = train_y[:-size_validation]

    # initialize parameters
    W1 = np.random.uniform(-0.05, 0.05, (hidden_size, pic_size))
    b1 = np.random.uniform(-0.05, 0.05, (hidden_size, 1))
    W2 = np.random.uniform(-0.05, 0.05, (num_classes, hidden_size))
    b2 = np.random.uniform(-0.05, 0.05,(num_classes, 1))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    # data processing in the neuron network
    epoch_num = 100
    for e in range(0,epoch_num):
        train_x, train_y = shuffle_files(train_x, train_y)
        # run the train files
        params, train_loss = run_train_files(train_x, train_y, params)
        # run the validation files
        validation_correct_average = run_validation_files(validation_x, validation_y, params)

        print("\n" + str(e))
        print("#train loss: " + str(train_loss))
        print("#validation: " + str(validation_correct_average))

    # run test file and write the precision of each picture to "test_y" file
    # file = open("test_y", "w")
    # for row in test_x:
    #     # norm x
    #     row = row / biggest_value_x
    #     # calculate the forward algorithm till the level of softmax function
    #     y_hat = fprop(row, train_y, params, 0)
    #     # choose the biggest value from the output vector
    #     file.write(str(y_hat.argmax()) + '\n')
    # file.close()


if __name__ == "__main__":
    main()