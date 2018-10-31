# trick list:
- user defined loss

  ```
  loss_less = 1
  loss_more = 10
  loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
  train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
  ```
- setting learning rate
```
# 设置总的迭代次数
TRAINING_STEPS = 100
# 设置控制当前迭代次数的step变量
global_step = tf.Variable(0)
# decay 由两个东西共同决定：一个是 0.96 ， 一个是 global_step
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

# 定义网络
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

# 定义优化方法，此时传入的是 LEARNING_RATE ， minimize 时要传入 step ， step 会自主更新
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    # 初始化
    sess.run(tf.global_variables_initializer())
    
    for i in range(TRAINING_STEPS):
    
        sess.run(train_op)
        if i % 10 == 0:
            # 每个10 次拿出learning rate 看看
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print "After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value)

```


- l2 normalization
【注】涉及到通过collection 控制loss 的技术
  ```
  # 为每个weight 定义一个loss

  def get_weight(shape, lambda1):
      var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
      tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
      return var


  # 构建网络，loss 自动生成 ， 并加入到‘loss’中
  # loss 组的访问方法为tf.get_collection('losses')

  x = tf.placeholder(tf.float32, shape=(None, 2))
  y_ = tf.placeholder(tf.float32, shape=(None, 1))
  sample_size = len(data)

  # 每层节点的个数
  layer_dimension = [2,10,5,3,1]

  n_layers = len(layer_dimension)

  cur_layer = x
  in_dimension = layer_dimension[0]

  # 循环生成网络结构
  for i in range(1, n_layers):
      out_dimension = layer_dimension[i]
      weight = get_weight([in_dimension, out_dimension], 0.003)
      bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
      cur_layer = tf.nn.elu(tf.matmul(cur_layer, weight) + bias)
      in_dimension = layer_dimension[i]

  y= cur_layer

  # 损失函数的定义。
  mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
  tf.add_to_collection('losses', mse_loss)
  loss = tf.add_n(tf.get_collection('losses'))

  ```




- moving avg
【略】

