from inception_v2 import stochastic_branch_fn

import tensorflow as tf
import numpy as np

def main():
    with tf.Session() as sess:
        s = [32, 10]
        preact = tf.constant(np.random.random(size=s), dtype=tf.float32)
        slope = tf.constant(1.0, dtype=tf.float32)
        prob = tf.nn.softmax(preact * slope, dim=1)
        samp = stochastic_branch_fn(preact, slope, True)
        
        randVec = tf.constant(np.random.random(size=s), dtype=tf.float32)
        loss1 = tf.reduce_sum(prob * randVec)
        loss2 = tf.reduce_sum(samp * randVec)
        grad1 = tf.gradients(loss1, preact)[0]
        grad2 = tf.gradients(loss2, preact)[0]

        tmp = np.zeros((32, 10))
        tmp2 = 0.0
        n = 100000
        for _ in xrange(n):
            prob_, samp_, grad1_, grad2_ = sess.run([prob, samp, grad1, grad2])
            tmp2 = max(tmp2, np.max(np.abs(grad1_ - grad2_)))
            tmp += samp_ - prob_
        tmp /= n
        print tmp2
        print np.max(np.abs(tmp))

if __name__ == '__main__':
    main()