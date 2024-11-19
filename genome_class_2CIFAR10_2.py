import numpy as np
import random as rand
import math
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Input, concatenate, Add, Conv2D, merge, GlobalMaxPooling2D, GlobalAveragePooling2D, Reshape, Lambda, Concatenate,Multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, SeparableConvolution2D, Conv2DTranspose,  AveragePooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from metrics import *
import keras.losses

keras.losses.dice_coef_loss = dice_coef_loss
import keras.metrics

keras.metrics.dice_coef = dice_coef
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal, VarianceScaling
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.optimizers import adam, rmsprop, adadelta, adagrad
import tensorflow as tf


class ModelMGPU(Model):
    def __init__(self, model, gpus):
        pmodel = multi_gpu_model(model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


class GenomeHandler:
    """
    Defines the configuration and handles the conversion and mutation of
    individual genomes. Should be created and passed to a `DEvol` instance.

    ---
    Genomes are represented as fixed-with lists of integers corresponding
    to sequential layers and properties. A model with 2 convolutional layers
    and 1 dense layer would look like:

    [<conv layer><conv layer><dense layer><optimizer>]

    The makeup of the convolutional layers and dense layers is defined in the
    GenomeHandler below under self.convolutional_layer_shape and
    self.dense_layer_shape. <optimizer> consists of just one property.
    """

    def __init__(self, max_block_num, max_filters,max_dense_layers,max_dense_nodes, n_classes,
                 input_shape, batch_normalization=True, dropout=True, max_pooling=True,
                 optimizers=None, activations=None, learningrates=None, batch_size=None, augsize=None, kernels=None):
        """
        Creates a GenomeHandler according

        Args:
            max_conv_layers: The maximum number of convolutional layers
            max_conv_layers: The maximum number of dense (fully connected)
                    layers, including output layer
            max_filters: The maximum number of conv filters (feature maps) in a
                    convolutional layer
            max_dense_nodes: The maximum number of nodes in a dense layer
            input_shape: The shape of the input
            n_classes: The number of classes
            batch_normalization (bool): whether the GP should include batch norm
            dropout (bool): whether the GP should include dropout
            max_pooling (bool): whether the GP should include max pooling layers
            optimizers (list): list of optimizers to be tried by the GP. By
                    default, the network uses Keras's built-in adam, rmsprop,
                    adagrad, and adadelta
            activations (list): list of activation functions to be tried by the
                    GP. By default, relu and sigmoid.
        """
        # if max_dense_layers < 1:
        #     raise ValueError(
        #         "At least one dense layer is required for softmax layer"
        #     )
        if max_filters > 0:
            filter_range_max = int(math.log(max_filters, 2)) + 1
        else:
            filter_range_max = 0
        self.optimizer = optimizers or [
            'adam',
            'rmsprop',
            'adagrad',
            'adadelta'
        ]
        self.kernel = kernels or [
            'RandomNormal',
            'RandomUniform',
            'TruncatedNormal',
            'VarianceScaling',
            'glorot_normal',
            'glorot_uniform',
            'he_normal',
            'he_uniform'
        ]        
        self.learningrate = learningrates or [
            0.1,
            0.01,
            0.001,
            0.0001,
        ]
        self.batchsize = batch_size or [
            4,
            8,
            16,
            

        ]
        self.augmentationsize = augsize or [
            8000,
            16000,
            32000,
            

        ]
        # self.conv = convs or [
        # 'Convolution1D',
        # 'Convolution2D',
        # ]
        self.activation = activations or [
            'relu',
            'sigmoid',
            'tanh',
            'elu',
        ]
        self.convolutional_layer_shape = [
            "active",
            "att",
            # "typeshortcon",
            "longcon",
            # "typelongcon",
            "conv num",
            "conv size",
            # "conv",
            "num filters",
            "batch normalization",
            "activation",
            "dropout",
            "pooling",
        ]
        self.dense_layer_shape = [
            "type",
            "num nodes",
            "batch normalization",
            "activation",
            "dropout",
        ]
        self.layer_params = {

            "active": [0, 1],
            "type": list(range(0, 3)),
            "att": list(range(0, 10)),
            # "typeshortcon": [0,1], # 0=elementwise sum, 1=concatenation
            "longcon": [0, 1],
            # "typelongcon":[0,1], # 0=elementwise sum, l=concatenation
            "conv num": list(range(1, 6)),
            # "conv": list(range(len(self.conv))),
            "conv size": [3, 5, 7],  # 3= 3*3 filter size, 5=5*5 filter size
            "num filters": [2 ** i for i in range(3, filter_range_max)],
            "num nodes": [2**i for i in range(6, int(math.log(max_dense_nodes, 2)) + 1)],
            "batch normalization": [0, (1 if batch_normalization else 0)],
            "activation": list(range(len(self.activation))),
            "dropout": [(i if dropout else 0) for i in range(7)],
            # "max pooling": list(range(3)) if max_pooling else 0,
            "pooling": [0, 1],  # 1=maxpooling, 0=averagepooling
        }

        self.convolution_layers = max_block_num
        self.convolution_layer_size = len(self.convolutional_layer_shape)
        self.dense_layers = max_dense_layers - 1 # this doesn't include the softmax layer, so -1
        self.dense_layer_size = len(self.dense_layer_shape)
        self.input_shape = input_shape
        self.n_classes = n_classes

    def convParam(self, i):
        key = self.convolutional_layer_shape[i]
        return self.layer_params[key]

    def denseParam(self, i):
        key = self.dense_layer_shape[i]
        return self.layer_params[key]


    def mutate(self, genome, num_mutations):
        num_mutations = np.random.choice(num_mutations)
        for i in range(num_mutations):
            index = np.random.choice(list(range(1, len(genome))))
            if index < self.convolution_layer_size * self.convolution_layers:
                if genome[index - index % self.convolution_layer_size]:
                    range_index = index % self.convolution_layer_size
                    choice_range = self.convParam(range_index)
                    genome[index] = np.random.choice(choice_range)
                elif rand.uniform(0, 1) <= 0.01:  # randomly flip deactivated layers
                    genome[index - index % self.convolution_layer_size] = 1
            elif index>=70 and index<75:
                offset = self.convolution_layer_size * self.convolution_layers
                new_index = (index - offset)
                #present_index = new_index - new_index % self.dense_layer_size
                #if genome[new_index + offset]:
                # range_index = new_index % self.dense_layer_size
                choice_range = self.denseParam(new_index)
                genome[index] = np.random.choice(choice_range)
                #elif rand.uniform(0, 1) <= 0.01:
                    #genome[present_index + offset] = 1
            elif index == 75:
                genome[index] = np.random.choice(list(range(len(self.optimizer))))

            elif index == 76:
                genome[index] = np.random.choice(list(range(len(self.learningrate))))

            elif index == 77:
                genome[index] = np.random.choice(list(range(len(self.batchsize))))
            elif index == 78:
                genome[index] = np.random.choice(list(range(len(self.augmentationsize))))
            elif index ==79:
                genome[index] = np.random.choice(list(range(len(self.kernel))))    

        return genome

    def conv_block(self, m, dim, acti, bn, att, dp, cn, cs, active, k):

        if active:
            if k== 0:
                 init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
            elif k == 1:
                init = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
            elif k == 2:
                init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
            elif k== 3:
                init = VarianceScaling(scale=1.0 / 9.0)
            elif k == 4:
                init = keras.initializers.glorot_normal(seed=None)
            elif k == 5:
                init = keras.initializers.glorot_uniform(seed=None)
            elif k == 6:
                init = keras.initializers.he_normal(seed=None)
            elif k == 7:
                init = keras.initializers.he_uniform(seed=None)
                
            
            n = Conv2D(dim, (cs, cs), activation=acti, padding='same', kernel_initializer=init)(m)
            n = BatchNormalization()(n) if bn else n
            # n = Dropout(float(dp / 20.0))(n)
            for j in range(cn - 1):
                n = Conv2D(dim, (cs, cs), activation=acti, padding='same', kernel_initializer=init)(n)
                n = BatchNormalization()(n) if bn else n
                # n = Dropout(float(dp / 20.0))(n)
            n = Dropout(float(dp / 10.0))(n)
            if att == 0:  # without attention
                return n
            elif att ==1:   #attention 1
                #   m is input     n    is output
                
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  #  input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1),  padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1),  padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                nn = n  
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Add()([n, m])
                return Concatenate(axis=3)([nn, n])
            elif att ==2:  #attention 4-1
                
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  # input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                    
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Add()([n, m])
                return n
            elif att==3: # attention 4-2
                nn = n
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Add()([nn, n])
                return n
            elif att==4:  # attention six
                
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  # input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                nn = n
                n = keras.layers.Add()([n, m])
                n = keras.layers.Activation(activation='relu')(n)
                n = Conv2D(int(max(dim1,dim2)), (3, 3), padding='same', kernel_initializer=init)(n)
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Multiply()([nn, n])
                return n
            elif att == 5: # attention eight
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  # input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                n = keras.layers.Add()([n, m])
                n = keras.layers.Activation(activation='relu')(n)
                n = Conv2D(int(max(dim1,dim2)), (3, 3), padding='same', kernel_initializer=init)(n)
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Multiply()([m, n])
                return n
            elif att == 6: # attention ten
                                
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  # input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                nn = n    
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Multiply()([m, n])
                n = keras.layers.Add()([n, nn, m])
                return n
            elif att == 7: # attention five
                nn = n
                dim_n = np.shape(n)  # output
                dim1 = dim_n[3]
                n = Conv2D(int(dim1), (3, 3), padding='same', kernel_initializer=init)(n)
                n = BatchNormalization()(n)
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Multiply()([n, nn])
                return n
            elif att == 8: #attention 11-2
            
                nn = n
                dim_n = np.shape(n)  # output
                dim1 = dim_n[3]
                n = Conv2D(int(dim1), (3, 3), padding='same', kernel_initializer=init)(n)
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Add()([n, nn])
                return n
            elif att ==9: #attention 12-2
                dim_n = np.shape(n)  # output
                dim_m = np.shape(m)  # input
                dim1 = dim_n[3]
                dim2 = dim_m[3]
                if dim_n[3] > dim_m[3]:
                    m = Conv2D(int(dim1), (1, 1), padding='same', kernel_initializer=init)(m)
                    m = BatchNormalization()(m) if bn else m

                elif dim_m[3] > dim_n[3]:
                    n = Conv2D(int(dim2), (1, 1), padding='same', kernel_initializer=init)(n)
                    n = BatchNormalization()(n) if bn else n
                nn = n
                n = keras.layers.Add()([n, m])
                n = keras.layers.Activation(activation='relu')(n)
                n = Conv2D(int(max(dim1,dim2)), (3, 3), padding='same', kernel_initializer=init)(n)
                n = keras.layers.Activation(activation='sigmoid')(n)
                n = keras.layers.Multiply()([m, n])
                n = keras.layers.Add()([n, m])
                return n



        else:
            return m

    def level_block(self, m, genome, depth, up, offset):

        if depth > 1:

            active = genome[offset]
            att = genome[offset + 1]  # short Connection
            # tsc = genome[offset + 2]  # type of shortcut connection, elementwise sum or concatenation
            lc = genome[offset + 2]  # long connection
            # tlc = genome[offset + 4]  # type of long conecction, elementwise sum or concatenation
            cn = genome[offset + 3]  # the number of convolution layers
            cs = genome[offset + 4]  # the size of filter 3*3 or 5*5
            dim = genome[offset + 5]  # the number of filters
            bn = genome[offset + 6]  # the Batch normalization
            ac = genome[offset + 7]  # Activation functions
            dp = genome[offset + 8]  # the dropout
            pl = genome[offset + 9]  # type of pooling, maxpooling=1 or average pooling=0
            if ac == 0:
                acti = 'relu'
            elif ac ==1 :
                acti = 'sigmoid'
            elif ac ==2 :
                acti = 'tanh'
            else:
                acti = 'elu'
            k = genome[79]    
            n = self.conv_block(m, dim, acti, bn, att, dp, cn, cs, active, k)
            if pl == 1 and active:
                m = MaxPooling2D()(n)
            elif pl == 0 and active:
                m = AveragePooling2D()(n)
            offset += self.convolution_layer_size
            # offset1=offset

            m = self.level_block(m, genome, depth - 1, up, offset)



        return m

    def EvoUNet(self, genome, depth, upconv=False):
        out_ch = 1

        print(upconv)
        img_shape = (32, 32, 3)
        i = Input(shape=img_shape)
        o = self.level_block(i, genome, depth, upconv, 0)
        densetype = genome[70]
        NON = genome[71]
        BN = genome[72]
        AF = genome [73]
        D = genome [74]

        if densetype==0:
             flat = Flatten()(o)
             if AF == 0:
                 acti = 'relu'
             elif AF == 1:
                 acti = 'sigmoid'
             elif AF == 2:
                 acti = 'tanh'
             else:
                 acti = 'elu'
             o = Dense(NON, activation=acti)(flat)
             o = BatchNormalization()(o) if BN else o
             o = Dropout(float(D / 10.0))(o)
        elif densetype ==1:
             o = GlobalAveragePooling2D()(o)
        else:
            o = GlobalMaxPooling2D()(o)
        o = Dense(10, activation='softmax')(o)
        return Model(inputs=i, outputs=o)

    def decode(self, genome):
        # if not self.is_compatible_genome(genome):
        #     raise ValueError("Invalid genome for specified configs")
        print(genome)
        model = self.EvoUNet(genome, 6, upconv=False)

        model.summary()
        pl_model = model
        #pl_model = ModelMGPU(model, gpus=4)
        op = self.optimizer[genome[75]]
        batch = self.batchsize[genome[77]]
        aug = self.augmentationsize[genome[78]]
        print(op)
        if op == 'adam':
            pl_model.compile(optimizer=adam(lr=self.learningrate[genome[76]]), loss='categorical_crossentropy',  metrics=['accuracy'])
        elif op == 'rmsprop':
            pl_model.compile(optimizer=rmsprop(lr=self.learningrate[genome[76]]),loss='categorical_crossentropy', metrics=['accuracy'])
        elif op == 'adadelta':

            pl_model.compile(optimizer=adadelta(lr=self.learningrate[genome[76]]), loss='categorical_crossentropy',metrics=['accuracy'])
        else:
            pl_model.compile(optimizer=adagrad(lr=self.learningrate[genome[76]]),loss='categorical_crossentropy', metrics=['accuracy'])

        return pl_model,model, batch, aug

    def genome_representation(self):
        encoding = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                encoding.append("Conv" + str(i) + " " + key)
        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                encoding.append("Dense" + str(i) + " " + key)
        encoding.append("Optimizer")
        encoding.append("Learning Rate")
        encoding.append("Batch Size")
        encoding.append("Augmentation Size")
        encoding.append("Initializer")

        return encoding

    def generate(self):
        genome = []
        for i in range(self.convolution_layers):
            for key in self.convolutional_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))

        for i in range(self.dense_layers):
            for key in self.dense_layer_shape:
                param = self.layer_params[key]
                genome.append(np.random.choice(param))

        genome.append(np.random.choice(list(range(len(self.optimizer)))))
        genome.append(np.random.choice(list(range(len(self.learningrate)))))
        genome.append(np.random.choice(list(range(len(self.batchsize)))))
        genome.append(np.random.choice(list(range(len(self.augmentationsize)))))
        genome.append(np.random.choice(list(range(len(self.kernel)))))
        genome[0] = 1
        # genome = [1, 10, 0, 1, 5, 8, 0, 2, 3, 0,
        #           1, 10, 1, 1, 7, 8, 1, 0, 5, 1,
        #           1, 0, 1, 1, 7, 8, 1, 0, 5, 1,
        #           0, 0, 1, 1, 5, 8, 1, 3, 3, 0,
        #           1, 3, 0, 3, 5, 8, 0, 2, 4, 0,
        #           1, 1, 0, 3, 7, 8, 0, 3, 5, 0,
        #           2, 64, 1, 3, 4,
        #           3, 1, 0, 0]

        # genome[40] = 1

        return genome

    # def is_compatible_genome(self, genome):
    #     expected_len = self.convolution_layers * self.convolution_layer_size \
    #                     + self.dense_layers * self.dense_layer_size + 1
    #     if len(genome) != expected_len:
    #         return False
    #     ind = 0
    #     for i in range(self.convolution_layers):
    #         for j in range(self.convolution_layer_size):
    #             if genome[ind + j] not in self.convParam(j):
    #                 return False
    #         ind += self.convolution_layer_size
    #     for i in range(self.dense_layers):
    #         for j in range(self.dense_layer_size):
    #             if genome[ind + j] not in self.denseParam(j):
    #                 return False
    #         ind += self.dense_layer_size
    #     if genome[ind] not in range(len(self.optimizer)):
    #         return False
    #     print("eightth")
    #     return True

    def best_genome(self, csv_path, metric='accuracy', include_metrics=True):
        best = max if metric is 'accuracy' else min
        col = -1 if metric is 'accuracy' else -2
        data = np.genfromtxt(csv_path, delimiter=",")
        row = list(data[:, col]).index(best(data[:, col]))
        genome = list(map(int, data[row, :-2]))
        if include_metrics:
            genome += list(data[row, -2:])

        return genome

    def decode_best(self, csv_path, metric='accuracy'):

        return self.decode(self.best_genome(csv_path, metric, False))
