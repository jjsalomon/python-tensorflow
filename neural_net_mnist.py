import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''

input > weight > hidden layer 1 (activation function) > weights > hidden layer 2
(activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimization function (optimizer) > minimize cost(AdamOptimizer...SGD, AdaGrad)

Backpropagation

feed forward + backprop = epoch

'''

# one_hot - one is on, the rest are off
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
# Layers
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
.
.
'''

n_classes = 10
# go through batches of 100 of features then manipulate
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')


def neural_network_model(data):
    # Creates a tensor of data using random numbers that are weights
    # (input data * weights) + bias
    # If all input data = 0, no neuron would ever fire, Bias is a parameter that adds a value so that some neurons can fire
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

    
    # Layer one
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    # Activation function
    l1 = tf.nn.relu(l1)

    # Layer two
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    # Activation function
    l2 = tf.nn.relu(l2)

    # Layer three
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    # Activation function
    l3 = tf.nn.relu(l3)

    # Output Layer
    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Learning rate = 0.001 default
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y} )

                epoch_loss += c

            print('Epoch',epoch,'Completed Out of',hm_epochs,'Loss:',epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)