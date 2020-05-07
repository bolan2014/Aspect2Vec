# -*- coding: utf-8 -*-


from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def fscore(preds, dtrain):
    label = dtrain.get_label()
    pred = [int(i >= 0.5) for i in preds]
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    F = 2*p*r / (p+r)
    return 'fscore', float(F)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def distance_acc(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def hamming_distance(vectors):
    x, y = vectors
    return K.maximum(K.sum(K.abs(K.sign(x) - K.sign(y)), axis=1, keepdims=True), K.epsilon())


def dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def cosine_distance(vectors):
    x, y = vectors
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x*y, axis=-1, keepdims=True)


def manhattan_distance(vectors):
    x, y = vectors
    return K.maximum(K.sum(K.abs(x - y), axis=1, keepdims=True), K.epsilon())


def judge_by_threshold(predict_res, threshold=0.5):
    result = []
    for probability in predict_res:
        if probability > threshold:
            result.append(1)
        else:
            result.append(0)
    return result


def get_metric_sk(true_label, pred_label):
    precision = precision_score(y_true=true_label, y_pred=pred_label)
    recall = recall_score(y_true=true_label, y_pred=pred_label)
    f1 = f1_score(y_true=true_label, y_pred=pred_label)
    acc = accuracy_score(y_true=true_label, y_pred=pred_label)

    return [precision, recall, f1, acc]


class CategoricalMetircs(Callback):
    def on_train_begin(self, logs=None):
        self.val_precision = []
        self.val_recall = []
        self.val_loss = []
        self.val_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        valid_result = self.model.predict([
            self.validation_data[0],
            self.validation_data[1],
            # self.validation_data[2],
            # self.validation_data[3]
        ]).ravel()
        valid_tag = judge_by_threshold(valid_result)
        valid_score = get_metric_sk(true_label=self.validation_data[2], pred_label=valid_tag)
        self.val_precision.append(valid_score[0])
        self.val_recall.append(valid_score[1])
        self.val_f1.append(valid_score[2])
        self.val_loss.append(valid_score[3])
        print("valid_metrics: p-%.4f r-%.4f f1-%.4f" % (valid_score[0], valid_score[1], valid_score[2]))
        print(max(valid_result), min(valid_result))
        return


class DistanceMetirc(Callback):
    def on_train_begin(self, logs=None):
        self.val_precision = []
        self.val_recall = []
        self.val_acc = []
        self.val_f1 = []

    def on_epoch_end(self, epoch, logs=None):
        valid_result = self.model.predict([
            self.validation_data[0],
            self.validation_data[1],
            # self.validation_data[2],
            # self.validation_data[3],
        ]).ravel()
        valid_tag = [int(y < 0.3) for y in valid_result]
        valid_score = get_metric_sk(true_label=self.validation_data[2], pred_label=valid_tag)
        self.val_precision.append(valid_score[0])
        self.val_recall.append(valid_score[1])
        self.val_f1.append(valid_score[2])
        self.val_acc.append(valid_score[3])
        print("valid_metrics: acc-{:.4f} p-{:.4f} r-{:.4f} f1-{:.4f}".format(valid_score[3], valid_score[0], valid_score[1], valid_score[2]))
        print("max distance: {:.4f}, min distance: {:.4f}".format(max(valid_result), min(valid_result)))
        return

