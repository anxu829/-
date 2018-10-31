# import package
```
import tensorflow as tf
from numpy.random import RandomState
```


# 0. 生成数据

```
rdm = RandomState(1)
X = rdm.rand(128,2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
```

# 1. 定义net

```
# 可控参数使用variable  ， 在循环中人为更替的使用placeholder
batch_size = 8
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)


```

# 2. 定义loss
```
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))

```

# 3. 定义 opt
```
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

```


# 4. 定义batch iteration：
## - 在sess 中的流程为： 使用dataloader 逐步的load 数据 -> 通过feed_dict 传入数据 -> run 优化节点
## - 使用了 sess 来初始化
## - 使用了 sess.run([train_step]) 来完成优化
## - 使用了 sess.run(cross_entropy) 来完成评估

```
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    # 输出目前（未经训练）的参数取值。
    print(sess.run(w1))
    print(sess.run(w2))
    print("\n")
    
    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % 128
        end = (i*batch_size) % 128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))
```
