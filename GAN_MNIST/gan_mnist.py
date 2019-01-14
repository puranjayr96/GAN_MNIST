import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")
# Discriminator function

def disc(x_image):

# First block
    dlayer1 = tf.layers.conv2d(inputs = x_image,filters= 32,kernel_size=[5,5],strides=1,padding="same",
                              kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                              activation=tf.nn.leaky_relu,bias_initializer=tf.initializers.truncated_normal(stddev=0),reuse=tf.AUTO_REUSE,name="dlayer1")
    dlayer1 = tf.layers.batch_normalization(inputs=dlayer1, epsilon=1e-5)
    dlayer1 = tf.layers.average_pooling2d(inputs=dlayer1,pool_size=[2,2],strides=2,padding="same")
# Second block
    dlayer2 = tf.layers.conv2d(inputs=dlayer1, filters=64, kernel_size=[5, 5], strides=1, padding="same",
                              kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                              activation=tf.nn.leaky_relu, bias_initializer=tf.initializers.truncated_normal(stddev=0),reuse=tf.AUTO_REUSE,name="dlayer2")
    dlayer2 = tf.layers.batch_normalization(inputs=dlayer2, epsilon=1e-5)
    dlayer2 = tf.layers.average_pooling2d(inputs=dlayer2, pool_size=[2, 2], strides=2, padding="same")
# Third block FCC
    dlayer3 = tf.layers.flatten(dlayer2)
    dlayer3 = tf.layers.dense(inputs=dlayer3,units = 1024,activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                             bias_initializer=tf.initializers.constant(0),reuse=tf.AUTO_REUSE,name="dlayer3")
    dlayer3 = tf.layers.batch_normalization(inputs=dlayer3, epsilon=1e-5)
# Fourth block FCC
    dlayer4 = tf.layers.dense(inputs=dlayer3, units=1,kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                             bias_initializer=tf.initializers.constant(0),reuse=tf.AUTO_REUSE,name="dlayer4")
    dlayer4 = tf.layers.batch_normalization(inputs = dlayer4,epsilon = 1e-5)
    return dlayer4

# Generator function
def gen(batch_size, z_dim):
    z = tf.truncated_normal([batch_size,z_dim],name='z')

# First block
    glayer1 = tf.layers.dense(inputs=z,units=3136,activation=tf.nn.leaky_relu,
                             kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                             bias_initializer=tf.initializers.truncated_normal(stddev=0),reuse=tf.AUTO_REUSE,name="glayer1")
    glayer1 = tf.layers.batch_normalization(inputs=glayer1,epsilon=1e-5)
    glayer1 = tf.reshape(glayer1,[-1,56,56,1])
# Second block
    glayer2 = tf.layers.conv2d(inputs=glayer1,filters=z_dim/2,kernel_size=[3,3],strides=2,padding='same',
                              activation=tf.nn.leaky_relu,
                              kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                              bias_initializer=tf.initializers.truncated_normal(stddev=0),reuse=tf.AUTO_REUSE,name="glayer2")
    glayer2 = tf.layers.batch_normalization(inputs=glayer2,epsilon=1e-5)
    glayer2 = tf.image.resize_images(glayer2,[56,56])
# Third block
    glayer3 = tf.layers.conv2d(inputs=glayer2, filters=z_dim / 4, kernel_size=[3, 3], strides=2, padding='same',
                               activation=tf.nn.leaky_relu,
                               kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                               bias_initializer=tf.initializers.truncated_normal(stddev=0),reuse=tf.AUTO_REUSE,name="glayer3")
    glayer3 = tf.layers.batch_normalization(inputs=glayer3, epsilon=1e-5)
    glayer3 = tf.image.resize_images(glayer3, [56, 56])
# Fourth block
    glayer4 = tf.layers.conv2d(inputs=glayer3,filters=1,kernel_size=[1,1],strides=2,padding='same',
                               activation=tf.sigmoid,kernel_initializer=tf.initializers.truncated_normal(stddev=0.02),
                               bias_initializer=tf.initializers.truncated_normal(stddev=0.02),reuse=tf.AUTO_REUSE,name="glayer4")
    return glayer4

sess = tf.InteractiveSession()

z_dim = 100
batch_size = 100
x_placeholder = tf.placeholder("float",[None,28,28,1],name='x_placeholder')

Gz = gen(batch_size,z_dim)

Dx = disc(x_placeholder)

Dg = disc(Gz)
#So, let’s first think about what we want out of our networks. We want the generator network to create
#images that will fool the discriminator. The generator wants the discriminator to output a 1 (positive example).
#Therefore, we want to compute the loss between the Dg and label of 1. This can be done through
#the tf.nn.sigmoid_cross_entropy_with_logits function. This means that the cross entropy loss will
#be taken between the two arguments. The "with_logits" component means that the function will operate
#on unscaled values. Basically, this means that instead of using a softmax function to squish the output
#activations to probability values from 0 to 1, we simply return the unscaled value of the matrix multiplication.
#Take a look at the last line of our discriminator. There's no softmax or sigmoid layer at the end.
#The reduce mean function just takes the mean value of all of the components in the matrixx returned
#by the cross entropy function. This is just a way of reducing the loss to a single scalar value,
#instead of a vector or matrix.
#https://datascience.stackexchange.com/questions/9302/the-cross-entropy-error-function-in-neural-networks

# Generator Loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

# Discriminator Loss
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg,labels=tf.fill([batch_size,1],0.2)))
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx,labels=tf.fill([batch_size,1],0.9)))
d_loss = d_loss_fake +d_loss_real

with tf.variable_scope(tf.get_variable_scope(),reuse=False) as scope:
    # Trainer for discriminator
    d_trainer_real = tf.train.AdamOptimizer(0.001).minimize(d_loss_real)
    d_trainer_fake = tf.train.AdamOptimizer(0.001).minimize(d_loss_fake)
    # Trainer for generator
    g_trainer = tf.train.AdamOptimizer(0.001).minimize(g_loss)

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real',d_loss_real)
tf.summary.scalar('Discriminator_loss_fake',d_loss_fake)

d_real_count_ph = tf.placeholder(tf.float32)
d_fake_count_ph = tf.placeholder(tf.float32)
g_count_ph = tf.placeholder(tf.float32)

tf.summary.scalar('d_real_count',d_real_count_ph)
tf.summary.scalar('d_fake_count',d_fake_count_ph)
tf.summary.scalar('g_count',g_count_ph)

d_on_generated = tf.reduce_mean(disc(gen(batch_size,z_dim)))
d_on_real = tf.reduce_mean(disc(x_placeholder))

tf.summary.scalar('d_on_generated_eval', d_on_generated)
tf.summary.scalar('d_on_real_eval', d_on_real)

img_tensorboard = gen(batch_size,z_dim)
tf.summary.image('Generated images',img_tensorboard,10)
merged = tf.summary.merge_all()
logdir = "tensorboard/gan"
writer = tf.summary.FileWriter(logdir,graph=sess.graph)
print(logdir)

saver = tf.train.Saver()

test_gen = gen(3,z_dim)
tes_disc = disc(x_placeholder)
test_gen2 = gen(10,z_dim)


gLoss = 0
dLossFake, dLossReal = 1,1
d_real_count,d_fake_count,g_count = 0,0,0

sess.run(tf.global_variables_initializer())

for i in range(50000):

    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
    if abs(dLossFake - gLoss) <= 0.1:
        print("i:",i)
        print("dLossFake",dLossFake)
        print("gLoss",gLoss)
        print("Dif:",abs(dLossFake-gLoss))
        _,dLossReal,dLossFake,gLoss = sess.run([d_trainer_fake,d_loss_real,d_loss_fake,g_loss],
                                               {x_placeholder: real_image_batch})

        _, dLossReal, dLossFake, gLoss = sess.run([g_trainer,d_loss_real,d_loss_fake,g_loss],
                                               {x_placeholder: real_image_batch})
        g_count+=1
        d_fake_count += 1
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                  {x_placeholder: real_image_batch})
        d_real_count += 1
    elif dLossFake < gLoss:
        _, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
                                                  {x_placeholder: real_image_batch})
        g_count += 1
    elif gLoss < dLossFake:
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
                                                  {x_placeholder: real_image_batch})
        d_fake_count += 1
        _, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
                                                  {x_placeholder: real_image_batch})
        d_real_count += 1
    if i % 10 == 0:
        real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size,28,28,1])
        summary = sess.run(merged,{x_placeholder:real_image_batch,d_real_count_ph:d_real_count,d_fake_count_ph:d_fake_count,g_count_ph:g_count})
        writer.add_summary(summary,i)
        d_real_count,d_fake_count,g_count = 0,0,0

    if i % 1000 == 0:
        images = sess.run(test_gen)
        d_result = sess.run(tes_disc,{x_placeholder:images})
        print("Training Step ",i," at: ",datetime.datetime.now())
        for j in range(3):
            print("Discriminator Classification:",d_result[j])
            img = images[j,:,:,0]
            plt.imshow(img.reshape([28,28]),cmap='Greys')
            plt.show()
    if i % 5000 == 0:
        save_path = saver.save(sess,"models/pretrained_gan.ckpt",global_step=i)
        print("saved to %s" % save_path)

test_images = sess.run(test_gen2)
test_eval = sess.run(disc(x_placeholder), {x_placeholder: test_images})

real_images = mnist.validation.next_batch(10)[0].reshape([10, 28, 28, 1])
real_eval = sess.run(disc(x_placeholder), {x_placeholder: real_images})

# Show discriminator's probabilities for the generated images,
# and display the images
for i in range(10):
    print(test_eval[i])
    plt.imshow(test_images[i, :, :, 0], cmap='Greys')
    plt.show()

# Now do the same for real MNIST images
for i in range(10):
    print(real_eval[i])
    plt.imshow(real_images[i, :, :, 0], cmap='Greys')
    plt.show()