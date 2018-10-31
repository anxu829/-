# 基本概念：
- graph
  - graph
  - tensor
- session
  - init_op = tf.global_variables_initializer()
  - sess.run(op)
  ```
  # 图：利于变量管理
  import tensorflow as tf

  g1 = tf.Graph()
  with g1.as_default():
      v = tf.get_variable("v", [1], initializer = tf.zeros_initializer()) # 设置初始值为0

  g2 = tf.Graph()
  with g2.as_default():
      v = tf.get_variable("v", [1], initializer = tf.ones_initializer())  # 设置初始值为1

  with tf.Session(graph = g1) as sess:
      tf.global_variables_initializer().run()
      with tf.variable_scope("", reuse=True):
          print(sess.run(tf.get_variable("v")))

  with tf.Session(graph = g2) as sess:
      tf.global_variables_initializer().run()
      with tf.variable_scope("", reuse=True):
          print(sess.run(tf.get_variable("v")))
  ```
- config
  ```
  config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  sess1 = tf.InteractiveSession(config=config)
  sess2 = tf.Session(config=config)
  ```
  
  
