import tensorflow as tf 
import numpy as np 
#creat data 
#function : y = 0.01x +0.3

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.01 + 0.3                                  #y = b+w x b==>biases w==>weights
 
### creat tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))      # 权重的初始值
biases = tf.Variable(tf.zeros([1]))                         # 偏差的初始值

y =Weights*x_data+biases                                    #定义方程

loss = tf.reduce_mean(tf.square(y-y_data))                  #偏差 这里是平方误差
optimizer = tf.train.GradientDescentOptimizer(0.5)          #优化器，这里是初级的梯度下降优化器，0.5是学习效率，一般是一个小于1的数。
train = optimizer.minimize(loss)                            #

init = tf.initialize_all_variables()                        #启动器

### creat tensorflow structure end ###

sess = tf.Session()                                        
#sess.run是无比重要的，相当于一个启动指针，指向需要启动的地方。
sess.run(init)                                              #very important 激活initial

for step in range (201):
    sess.run(train)
    #每隔着20步就输出一次 Weights和biases 
    if step % 20 == 0: 
        print(step,sess.run(Weights),sess.run(biases)) 
        