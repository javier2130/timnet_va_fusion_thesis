"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
Modernized for TensorFlow >= 2.16 / Keras 3
"""
import numpy as np
import os
import tensorflow as tf
from keras.optimizers import Adam
from keras import callbacks
from keras.layers import Layer, Dense, Input
from keras.models import Model
from sklearn.metrics import confusion_matrix
from Common_Model import Common_Model
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import datetime
import pandas as pd
import copy

from TIMNET import TIMNET


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class WeightLayer(Layer):
    def __init__(self, **kwargs):
        super(WeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightLayer, self).build(input_shape)

    def call(self, x):
        tempx = tf.transpose(x, [0, 2, 1])
        x = tf.matmul(tempx, self.kernel)
        x = tf.squeeze(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# ============================================================
# [TESIS] VAFusionLayer — capa de fusión con softmax-attention
# (se aplica solo softmax)
# ============================================================
class VAFusionLayer(Layer):
    """Temporal fusion with softmax-normalized attention over scales."""
    def __init__(self, **kwargs):
        super(VAFusionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(VAFusionLayer, self).build(input_shape)

    def call(self, x):
        attn = tf.nn.softmax(self.kernel, axis=0)
        tempx = tf.transpose(x, [0, 2, 1])
        x = tf.matmul(tempx, attn)
        x = tf.squeeze(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])
# ============================================================
# [/TESIS] — desde aquí continúa código heredado de timnet
# ============================================================


class TIMNET_Model(Common_Model):
    def __init__(self, args, input_shape, class_label, **params):
        super(TIMNET_Model, self).__init__(**params)
        self.args = args
        self.data_shape = input_shape
        self.num_classes = len(class_label)
        self.class_label = class_label
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        print("TIMNET MODEL SHAPE:", input_shape)

    def create_model(self):
        self.inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size,
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation=self.args.activation,
                                return_sequences=True,
                                name='TIMNET')(self.inputs)

        self.decision = WeightLayer()(self.multi_decision)
        self.predictions = Dense(self.num_classes, activation='softmax')(self.decision)
        self.model = Model(inputs=self.inputs, outputs=self.predictions)

        self.model.compile(loss="categorical_crossentropy",
                           optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
                           metrics=['accuracy'])
        print("Temporal create success!")

    def train(self, x, y):

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        i = 1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        for train, test in kfold.split(x, y):
            self.create_model()
            y_train = smooth_labels(copy.deepcopy(y[train]), 0.1)
            folder_address = filepath + self.args.data + "_" + str(self.args.random_seed) + "_" + now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".weights.h5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=self.args.save_best_only, monitor='val_loss')
            max_acc = 0
            best_eva_list = []
            h = self.model.fit(x[train], y_train, validation_data=(x[test], y[test]), batch_size=self.args.batch_size, epochs=self.args.epoch, verbose=1, callbacks=[checkpoint])
            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:", str(round(avg_accuracy * 10000) / 100 / i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label))

        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold
        writer = pd.ExcelWriter(resultpath + self.args.data + '_' + str(self.args.split_fold) + 'fold_' + str(round(self.acc * 10000) / 100) + "_" + str(self.args.random_seed) + "_" + now_time + '.xlsx')
        for i, item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j, l in enumerate(item):
                temp[self.class_label[j]] = item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer, sheet_name=str(i))

            df = pd.DataFrame(self.eva_matrix[i]).transpose()
            df.to_excel(writer, sheet_name=str(i) + "_evaluate")
        writer.close()

        tf.keras.backend.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True

    def test(self, x, y, path):
        i = 1
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        x_feats = []
        y_labels = []
        for train, test in kfold.split(x, y):
            self.create_model()
            weight_path = path + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".weights.h5"
            self.model.fit(x[train], y[train], validation_data=(x[test], y[test]), batch_size=64, epochs=0, verbose=0)
            self.model.load_weights(weight_path)
            best_eva_list = self.model.evaluate(x[test], y[test])
            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[1]
            print(str(i) + '_Model evaluation: ', best_eva_list, "   Now ACC:", str(round(avg_accuracy * 10000) / 100 / i))
            i += 1
            y_pred_best = self.model.predict(x[test])
            self.matrix.append(confusion_matrix(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1)))
            em = classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test], axis=1), np.argmax(y_pred_best, axis=1), target_names=self.class_label))
            caps_layer_model = Model(inputs=self.model.input,
            outputs=self.model.get_layer(index=-2).output)
            feature_source = caps_layer_model.predict(x[test])
            x_feats.append(feature_source)
            y_labels.append(y[test])
        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold
        return x_feats, y_labels

    def train_tsne(self, x, y):
        """Train a single model with 80/20 split for t-SNE visualization."""
        filepath = self.args.model_path
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')

        x_train, x_test, y_train_raw, y_test = train_test_split(
            x, y, test_size=0.2, random_state=self.args.random_seed, stratify=np.argmax(y, axis=1)
        )

        self.create_model()
        y_train_smooth = smooth_labels(copy.deepcopy(y_train_raw), 0.1)

        folder_address = filepath + self.args.data + "_tsne_" + str(self.args.random_seed) + "_" + now_time
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)

        weight_path = folder_address + '/weights_best.weights.h5'
        checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')

        self.model.fit(x_train, y_train_smooth, validation_data=(x_test, y_test),
                       batch_size=self.args.batch_size, epochs=self.args.epoch, verbose=1, callbacks=[checkpoint])
        self.model.load_weights(weight_path)

        # Evaluate
        results = self.model.evaluate(x_test, y_test)
        print(f"Test evaluation - Loss: {results[0]:.4f}, ACC: {results[1]:.4f}")

        y_pred = self.model.predict(x_test)
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1), target_names=self.class_label))

        # Extract features for t-SNE from ALL data
        feat_model = Model(inputs=self.model.input, outputs=self.model.get_layer(index=-2).output)
        learned_feats = feat_model.predict(x)
        raw_feats = x.reshape(x.shape[0], -1)
        labels = np.argmax(y, axis=1)

        print(f"Saved weights to: {folder_address}")
        print(f"t-SNE features extracted from ALL {x.shape[0]} samples")
        return raw_feats, learned_feats, labels

    # ============================================================
    # [TESIS] Métodos V-A (multi-tarea clasificación + regresión V-A).
    # Todo lo que sigue hasta el final del archivo es aporte de la tesis:
    #   - train_va_tsne: entrena modelo V-A y extrae features de va_fusion
    #   - create_va_model: arquitectura dual-head (class + VA)
    #   - va_train: K-fold multi-tarea + circumplex V-A
    # ============================================================
    def train_va_tsne(self, x, y):
        """Train V-A model with 80/20 split and extract VAFusionLayer features for t-SNE."""
        from va_config import labels_to_va_targets

        tf.keras.backend.clear_session()

        filepath = self.args.model_path
        if not os.path.exists(filepath):
            os.mkdir(filepath)

        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')

        y_va = labels_to_va_targets(y, self.class_label)

        x_train, x_test, y_train_raw, y_test, y_train_va, y_test_va = train_test_split(
            x, y, y_va, test_size=0.2, random_state=self.args.random_seed, stratify=np.argmax(y, axis=1)
        )

        self.create_va_model()
        y_train_smooth = smooth_labels(copy.deepcopy(y_train_raw), 0.1)

        folder_address = filepath + self.args.data + "_va_tsne_" + str(self.args.random_seed) + "_" + now_time
        if not os.path.exists(folder_address):
            os.mkdir(folder_address)

        weight_path = folder_address + '/weights_best.weights.h5'
        checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')

        self.model.fit(
            x_train,
            {'class_output': y_train_smooth, 'va_output': y_train_va},
            validation_data=(x_test, {'class_output': y_test, 'va_output': y_test_va}),
            batch_size=self.args.batch_size,
            epochs=self.args.epoch,
            verbose=1,
            callbacks=[checkpoint]
        )
        self.model.load_weights(weight_path)

        class_pred, va_pred = self.model.predict(x_test)
        acc = np.mean(np.argmax(class_pred, axis=1) == np.argmax(y_test, axis=1))
        print(f"V-A Test ACC: {acc:.4f}")
        print(classification_report(np.argmax(y_test, axis=1), np.argmax(class_pred, axis=1), target_names=self.class_label))

        feat_model = Model(inputs=self.model.input, outputs=self.model.get_layer('va_fusion').output)
        va_feats = feat_model.predict(x)
        labels = np.argmax(y, axis=1)

        _, va_pred_all = self.model.predict(x)

        print(f"Saved weights to: {folder_address}")
        print(f"V-A t-SNE features extracted from all {x.shape[0]} samples")
        return va_feats, labels, va_pred_all

    def create_va_model(self):
        """Build dual-head model: classification + V-A regression."""
        from va_losses import combined_ccc_loss, ccc_metric_valence, ccc_metric_arousal

        self.inputs = Input(shape=(self.data_shape[0], self.data_shape[1]))
        self.multi_decision = TIMNET(nb_filters=self.args.filter_size,
                                kernel_size=self.args.kernel_size,
                                nb_stacks=self.args.stack_size,
                                dilations=self.args.dilation_size,
                                dropout_rate=self.args.dropout,
                                activation=self.args.activation,
                                return_sequences=True,
                                name='TIMNET')(self.inputs)

        self.shared_features = VAFusionLayer(name='va_fusion')(self.multi_decision)

        self.class_output = Dense(self.num_classes, activation='softmax', name='class_output')(self.shared_features)
        self.va_output = Dense(2, activation='tanh', name='va_output')(self.shared_features)

        self.model = Model(inputs=self.inputs, outputs=[self.class_output, self.va_output])

        self.model.compile(
            optimizer=Adam(learning_rate=self.args.lr, beta_1=self.args.beta1, beta_2=self.args.beta2, epsilon=1e-8),
            loss={
                'class_output': 'categorical_crossentropy',
                'va_output': combined_ccc_loss,
            },
            loss_weights={
                'class_output': self.args.lambda_ce,
                'va_output': self.args.lambda_v + self.args.lambda_a,
            },
            metrics={
                'class_output': ['accuracy'],
                'va_output': [ccc_metric_valence, ccc_metric_arousal],
            }
        )
        print("VA Dual-Head model created!")

    def va_train(self, x, y):
        """Multi-task training with V-A fusion."""
        from va_config import labels_to_va_targets

        filepath = self.args.model_path
        resultpath = self.args.result_path

        if not os.path.exists(filepath):
            os.mkdir(filepath)
        if not os.path.exists(resultpath):
            os.mkdir(resultpath)

        y_va = labels_to_va_targets(y, self.class_label)

        i = 1
        now = datetime.datetime.now()
        now_time = datetime.datetime.strftime(now, '%Y-%m-%d_%H-%M-%S')
        kfold = KFold(n_splits=self.args.split_fold, shuffle=True, random_state=self.args.random_seed)
        avg_accuracy = 0
        avg_loss = 0
        all_va_preds = []
        all_va_true = []
        all_labels = []

        for train, test in kfold.split(x, y):
            self.create_va_model()
            y_train_class = smooth_labels(copy.deepcopy(y[train]), 0.1)
            y_train_va = y_va[train]
            y_test_class = y[test]
            y_test_va = y_va[test]

            folder_address = filepath + self.args.data + "_va_" + str(self.args.random_seed) + "_" + now_time
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address + '/' + str(self.args.split_fold) + "-fold_weights_best_" + str(i) + ".weights.h5"
            checkpoint = callbacks.ModelCheckpoint(weight_path, verbose=1, save_weights_only=True, save_best_only=True, monitor='val_loss')

            self.model.fit(
                x[train],
                {'class_output': y_train_class, 'va_output': y_train_va},
                validation_data=(x[test], {'class_output': y_test_class, 'va_output': y_test_va}),
                batch_size=self.args.batch_size,
                epochs=self.args.epoch,
                verbose=1,
                callbacks=[checkpoint]
            )
            self.model.load_weights(weight_path)

            class_pred, va_pred = self.model.predict(x[test])
            acc = np.mean(np.argmax(class_pred, axis=1) == np.argmax(y_test_class, axis=1))
            avg_accuracy += acc

            print(f"{i}_Fold ACC: {acc:.4f}   Now AVG ACC: {avg_accuracy / i:.4f}")
            print(classification_report(np.argmax(y_test_class, axis=1), np.argmax(class_pred, axis=1), target_names=self.class_label))

            self.matrix.append(confusion_matrix(np.argmax(y_test_class, axis=1), np.argmax(class_pred, axis=1)))
            em = classification_report(np.argmax(y_test_class, axis=1), np.argmax(class_pred, axis=1), target_names=self.class_label, output_dict=True)
            self.eva_matrix.append(em)

            all_va_preds.append(va_pred)
            all_va_true.append(y_test_va)
            all_labels.append(np.argmax(y_test_class, axis=1))
            i += 1

        print("Average ACC:", avg_accuracy / self.args.split_fold)
        self.acc = avg_accuracy / self.args.split_fold

        # Save results Excel
        writer = pd.ExcelWriter(resultpath + self.args.data + '_VA_' + str(self.args.split_fold) + 'fold_' + str(round(self.acc * 10000) / 100) + "_" + str(self.args.random_seed) + "_" + now_time + '.xlsx')
        for idx, item in enumerate(self.matrix):
            temp = {}
            temp[" "] = self.class_label
            for j, l in enumerate(item):
                temp[self.class_label[j]] = item[j]
            data1 = pd.DataFrame(temp)
            data1.to_excel(writer, sheet_name=str(idx))
            df = pd.DataFrame(self.eva_matrix[idx]).transpose()
            df.to_excel(writer, sheet_name=str(idx) + "_evaluate")
        writer.close()

        # Generate V-A visualization
        all_va_preds = np.concatenate(all_va_preds, axis=0)
        all_va_true = np.concatenate(all_va_true, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        from va_visualization import plot_va_circumplex
        va_plot_path = resultpath + f'va_circumplex_{self.args.data}_{now_time}.png'
        plot_va_circumplex(all_va_true, all_va_preds, all_labels, self.class_label, self.args.data, va_plot_path)

        tf.keras.backend.clear_session()
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0
        self.trained = True
