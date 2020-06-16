import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print('train',mnist.train.num_examples,
#       ',validation',mnist.validation.num_examples,
#       ',test',mnist.test.num_examples)

def plot_image(image):
    plt.imshow(image.reshape(28,28),cmap='binary')
    plt.show()

# plot_image(mnist.train.images[0])
np.argmax(mnist.train.labels[0])

def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        
        ax.imshow(np.reshape(images[idx],(28, 28)), 
                  cmap='binary')
            
        title= "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

# plot_images_labels_prediction(mnist.validation.images,
#                               mnist.validation.labels,[],0)
# plot_images_labels_prediction(mnist.test.images,
#                               mnist.test.labels,[],0)

# batch_images_xs, batch_labels_ys = \
#     mnist.train.next_batch(batch_size=100)
    
# print(len(batch_images_xs),
#       len(batch_labels_ys))

def layer(output_dim,input_dim,inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

# 建立输入层
x = tf.placeholder("float", [None, 784])

# 建立隐藏层h1
h1=layer(output_dim=256,input_dim=784,
         inputs=x ,activation=tf.nn.relu)  

# 建立输出层
y_predict=layer(output_dim=10,input_dim=256,
                    inputs=h1,activation=None)
# 建立训练数据 label 真实值的 placeholder
y_label = tf.placeholder("float", [None, 10])

# 定义 loss function

loss_function = tf.reduce_mean(
                  tf.nn.softmax_cross_entropy_with_logits
                         (logits=y_predict , 
                          labels=y_label))

# 选择 optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.001) \
                    .minimize(loss_function)
# 计算每一项数据是否正確预测
# tf.equal,相等返回1，不等返回0
correct_prediction = tf.equal(tf.argmax(y_label  , 1),
                              tf.argmax(y_predict, 1))

# 将計算预测正确结果，加权平均
# 先使用 tf.cast 转化为 float，再使用 tf.reduce_mean 将所有值平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples/batchSize)
epoch_list=[];loss_list=[];accuracy_list=[]
from time import time
startTime=time()
sess = tf.Session() # 	计算时间
sess.run(tf.global_variables_initializer())

# for each epoch
for epoch in range(trainEpochs):
    # for each mini-batch
    for i in range(totalBatchs):
        # 调用读mini-batch的函数
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        # 训练优化器，最小化损失函数，输入x和y
        sess.run(optimizer,feed_dict={x: batch_x,y_label: batch_y})
    # 在验证集上计算精度和误差
    loss,acc = sess.run([loss_function,accuracy],
                        feed_dict={x: mnist.validation.images, 
                                   y_label: mnist.validation.labels})
    # 显示训练结果并存入列表
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(acc)    
    print("Train Epoch:", '%02d' % (epoch+1), "Loss=", \
                "{:.9f}".format(loss)," Accuracy=",acc)
                
# 计算训练的时间 
duration =time()-startTime
print("Train Finished takes:",duration)

%matplotlib inline
import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(5,3)
plt.plot(epoch_list, loss_list, label = 'loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'])
plt.show()



import matplotlib.pyplot as plt
import numpy as np
def plot_images_labels_prediction(images,labels,
                                  prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        
        ax.imshow(np.reshape(images[idx],(28, 28)), 
                  cmap='binary')
            
        title= "label=" +str(np.argmax(labels[idx]))
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx]) 
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()

plot_images_labels_prediction(mnist.test.images,
                              mnist.test.labels,
                              prediction_result,0)






