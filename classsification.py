import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('mnist_data',one_hot=True)


def add_layer(inputs,in_size,out_size,activation_function=None):      
    #加层函数，输入，，，激励函数（None指的是没有）

    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #Weights 是一个in_size行 out_size列的矩阵

    biases =tf.Variable(tf.zeros([1,out_size])+0.1)
    #biase 是一个列表，1行，out_size列，加0.1是因为biase推荐值不为0
    #注意，这里加0.1是在tf.zeros做好以后加，不可以在里面加因为list不能加上一个float，加在外面的意思是list里面的所有元素都加0.1

    Wx_plus_b = tf.matmul(inputs,Weights) +biases
    #Wx_plus_b储存的是尚未被激活的值
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs =activation_function(Wx_plus_b)
    return outputs

#计算准确度的功能：
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.arg_max(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict = {xs:v_xs,ys:v_ys})
    return result




# define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])         #每张图片784个数据点 28*28
ys = tf.placeholder(tf.float32,[None,10])          #预测的是0-9这10个数字，所以预测值为10


# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax) #softmax 是用于分类的一种激励函数

# the error between prediction and real data
# the loss(cross_entropy)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices = [1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

#初始化变量
sess.run(tf.initialize_all_variables())


#
for i in range(1000):
    #使用SGD（stochastic gradient decent）法分割数据，每次仅学习100个，提高学习效率
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i %25 ==0:
        #输出准确率
        print(compute_accuracy(mnist.test.images,mnist.test.labels))




