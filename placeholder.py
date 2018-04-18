import tensorflow as tf 

input1 = tf.placeholder(tf.float32)         #预定义一个float32类型的预定义量
input2 = tf.placeholder(tf.float32)         

output = tf.multiply(input1,input2)         #做乘法

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))     #填充预定义量用的是字典
    #填充字典的名称即为预定义量的名称，字典key就是输入量的值。