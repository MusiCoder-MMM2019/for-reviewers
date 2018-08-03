import tensorflow as tf
import transform

class textureTransferTester:

    def __init__(self, session, content_image, model_path):
        self.sess = session
        self.x0 = content_image
        self.model_path = model_path
        self.transform = transform.Transform()
        self._build_graph()

    def _build_graph(self):

        self.x = tf.placeholder(tf.float32, shape=self.x0.shape, name='input')
        self.xi = tf.expand_dims(self.x, 0) 

        self.y_hat = self.transform.net(self.xi/255.0)
        self.y_hat = tf.squeeze(self.y_hat)
        self.y_hat = tf.clip_by_value(self.y_hat, 0., 255.)

    def test(self):

        self.sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)

        output = self.sess.run(self.y_hat, feed_dict={self.x: self.x0})

        return output





