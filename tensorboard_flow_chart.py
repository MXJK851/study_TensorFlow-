import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 


def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):      
    #加层函数，输入，，，激励函数（None指的是没有）
    layer_name = 'layer%s'%n_layer
    with tf.name_scope(layer_name): #这个是大layer层的图层
        with tf.name_scope("weights"):  #这个是weight层的图层
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
            #Weights 是一个in_size行 out_size列的矩阵

            #histogram_summary  ==> weights
            tf.summary.histogram(layer_name+'/weights',Weights)


        with tf.name_scope("biases"):  #这个是biases层的图层
            biases =tf.Variable(tf.zeros([1,out_size])+0.1,name = "b")
        #biase 是一个列表，1行，out_size列，加0.1是因为biase推荐值不为0
        #注意，这里加0.1是在tf.zeros做好以后加，不可以在里面加因为list不能加上一个float，加在外面的意思是list里面的所有元素都加0.1

            #histogram_summary ==> biases
            tf.summary.histogram(layer_name+'/biases',biases)


        Wx_plus_b = tf.matmul(inputs,Weights) +biases
        #Wx_plus_b储存的是尚未被激活的值
        
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs =activation_function(Wx_plus_b)

        #histogram_summary ==> outputs
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs




# 开始建立一个神经网络模型
# 建造输入的标签和量，还有噪音

x_data = np.linspace(-1,3,300)[:,np.newaxis]       #这个x数据有300行
noise = np.random.normal(0,0.6,x_data.shape)      #噪点，更像真实数据，用了一个normal分布，方差0.05，x_data.shape是让噪音变成x的格式
y_data = np.square(x_data) -0.5 +noise             #数据的标签，这里用了平方

#placeholder 先hold住x和y的位置，为了后期的格式化输入，这里注意必须给变量一个定义，比如定义成float32型
with tf.name_scope("inputs"):#这个是input层的图层
    xs = tf.placeholder(tf.float32,[None,1],name = "x_input")
    ys = tf.placeholder(tf.float32,[None,1],name = "y_input")

l1 = add_layer(xs,1,20,n_layer=1,activation_function=tf.nn.sigmoid)     #含义是输入数据为xs，该输入数据维度为1，隐藏层有10个神经元，因此输出维度为10激活函数为relu
#！！！！这里需要注意一般来说要是出现问题，无法预测，一般是激活函数有点水，可以加换一个激活函数relu和sigmoid等等都是可以的，也可以去官网看。
prediction = add_layer(l1,20,1,n_layer=2,activation_function=None)   #输入数据为隐藏层l1输出的数据，该数据由于是10个神经元发出，因此数据维度为10，而输出的为y所以输出唯独为1

#计算loss，这里用的是均方差误差mean-square error, MSE
with tf.name_scope("loss"):  #这个是loss层的图层
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction,name = "square"),reduction_indices = [1],name="sum"),name="mean")
    
    
    #histogram_summary ==> loss
    tf.summary.scalar('loss',loss)

#进行train，使用梯度下降优化器（学习率为0.1，学习率可以取小于1的一个值），目的是减少loss
with tf.name_scope("train"):  #这个是train层的图层
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化所有变量
with tf.name_scope("inputs"):  #这个是input层的图层
    init = tf.initialize_all_variables()

#定义运行函数
sess = tf.Session()
sess.run(init)

# 出流程图
merged = tf.summary.merge_all()             #总结所有的merge,从而输出成histogram图以及一些数据分析图
writer = tf.summary.FileWriter("./",sess.graph)

#开始1000次的运行，feed_dict 是把值赋给placeholder的变量。每次都run一下train_step并在每50次时候输出一下loss
for i in range(1000):
    #学习1000次
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    #每10次执行一次统计histogram以及输出一次loss的具体数值
    if i%10 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        writer.add_summary(result,i)


#把出来的文件放到文件夹log中
#输入：tensorboard --logdir 'log/' 并在浏览器中浏览
