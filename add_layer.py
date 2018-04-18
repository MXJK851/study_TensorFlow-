import tensorflow as tf 

def add_layer(inputs,in_size,out_size,activation_function=None):      
    #加层函数，输入，，，激励函数（None指的是没有）

    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #Weights 是一个in_size行 out_size列的矩阵

    biases =tf.Variable(tf.zeros([1,out_size]+0.1))
    #biase 是一个列表，1行，out_size列，加0.1是因为biase推荐值不为0

    Wx_plus_b = tf.matmul(inputs,Weights) +biases
    #Wx_plus_b储存的是尚未被激活的值
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs =activation_function(Wx_plus_b)
    return outputs