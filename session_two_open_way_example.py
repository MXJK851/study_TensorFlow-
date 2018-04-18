#3 3    *   2
#           2
import tensorflow as tf 

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])

product = tf.matmul(matrix1,matrix2)      #matrix multiply 等于 numpy中的np.dot(m1,m2) matlab 中的m1.*m2

#method1
sess = tf.Session()                       #定义session
result1 = sess.run(product)                #在该次sess内run一下product
print(result1)                             #显示结果
sess.close()                              #关闭该次sess

#method2
with tf.Session() as sess:                  #with语句，打开sess 结束with语句就关闭了sess
    result2 = sess.run(product)
    print(result2) 