import tensorflow as tf 

state = tf.Variable(0,name="counter")              #不许定义成变量，他才是变量，第一个0是他的值，counter是该变量的名称属性


print(state.name)                                  #看下该变量的名字

one =tf.constant(1)                                #定义一个常量1

new_value = tf.add(state,one)
update =tf.assign(state,new_value)                 #这是一个功能，sess.run(update)意义为讲new_value赋值到state上
#must have if define variables
init =tf.global_variables_initializer()                 #初始化所有变量

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))