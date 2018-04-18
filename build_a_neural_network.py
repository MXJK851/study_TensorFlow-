import tensorflow as tf 
import numpy as np 

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



# 开始建立一个神经网络模型
# 建造输入的标签和量，还有噪音

x_data = np.linspace(-1,1,300)[:,np.newaxis]       #这个x数据有300行
noise = np.random.normal(0,0.05,x_data.shape)      #噪点，更像真实数据，用了一个normal分布，方差0.05，x_data.shape是让噪音变成x的格式
y_data = np.square(x_data) -0.5 +noise             #数据的标签，这里用了平方

#placeholder 先hold住x和y的位置，为了后期的格式化输入，这里注意必须给变量一个定义，比如定义成float32型
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(xs,1,20,activation_function=tf.nn.relu)     #含义是输入数据为xs，该输入数据维度为1，隐藏层有10个神经元，因此输出维度为10激活函数为relu
prediction = add_layer(l1,20,1,activation_function=None)   #输入数据为隐藏层l1输出的数据，该数据由于是10个神经元发出，因此数据维度为10，而输出的为y所以输出唯独为1

#计算loss，这里用的是均方差误差mean-square error, MSE,reduction_indices = [1]按行求和，reduction_indices = [0]按列求和
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices = [1]))

#进行train，使用梯度下降优化器（学习率为0.1，学习率可以取小于1的一个值），目的是减少loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化所有变量
init = tf.initialize_all_variables()

#定义运行函数
sess = tf.Session()
#运行初始化
sess.run(init)

#开始1000次的运行，feed_dict 是把值赋给placeholder的变量。每次都run一下train_step并在每50次时候输出一下loss
for i in range(1000000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))