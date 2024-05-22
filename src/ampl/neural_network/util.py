from ampl.enums import NNActivation, NNOptimizers


class Util(object):
    @staticmethod
    def get_keras_activation(activation: NNActivation = NNActivation.RELU):
        tf = Util.load_tensorflow()

        if isinstance(activation, str):
            activation = NNActivation(activation)

        activations = {
            NNActivation.RELU: tf.nn.relu,
            NNActivation.SOFTMAX: tf.nn.softmax,
            NNActivation.TANH: tf.nn.tanh,
            NNActivation.SIGMOID: tf.nn.sigmoid,
            NNActivation.SWISH: tf.nn.swish,
            NNActivation.ELU: tf.nn.elu,
            NNActivation.SELU: tf.nn.selu,
        }
        return activations.get(activation, activations[NNActivation.RELU])

    @staticmethod
    def get_keras_optimizer(lr: float = 0.1, my_optimizer: NNOptimizers = NNOptimizers.ADAM):
        """
                Get the optimizer used in the best model trial. From time to time new optimizers will be released
                by Keras. If you would like to use an optimizer then add it to the opts object. On error the default Optimizer
                is Adam

                :param my_optimizer:
                :param lr: Learning rate.
                :type lr: float
                :return: Keras optimizer
                :rtype: _type_
                """

        if isinstance(my_optimizer, str):
            my_optimizer = NNOptimizers(my_optimizer)

        tf = Util.load_tensorflow()

        optimizers = {  #
            NNOptimizers.ADAM: tf.keras.optimizers.Adam(learning_rate=lr),
            NNOptimizers.SGD: tf.keras.optimizers.SGD(learning_rate=lr),
            NNOptimizers.RMSPROP: tf.keras.optimizers.RMSprop(learning_rate=lr),
            NNOptimizers.NADAM: tf.keras.optimizers.Nadam(learning_rate=lr),
            NNOptimizers.ADADELTA: tf.keras.optimizers.Adadelta(learning_rate=lr),
            NNOptimizers.ADAGRAD: tf.keras.optimizers.Adagrad(learning_rate=lr),
            NNOptimizers.ADAMAX: tf.keras.optimizers.Adamax(learning_rate=lr)
        }
        return optimizers.get(NNOptimizers(my_optimizer), optimizers[NNOptimizers.ADAM])

    @staticmethod
    def load_tensorflow():
        """
        Lazy loading tensorflow
        :return: tensorflow module
        """
        import os
        import sys

        if sys.platform == 'darwin':
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['LANG'] = "en_US.UTF-8"
        os.environ['LANGUAGE'] = "en_US.UTF-8"
        os.environ['LC_ALL'] = "en_US.UTF-8"

        import tensorflow as tf
        return tf
