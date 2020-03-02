

Tensorflow:

    y_pred: [N, >1]    y_true: [N], [N,1]
        sparse_categorical_cross_entropy
    y_pred: [N, >1]    y_true: [N, >1]
        categorical_cross_entropy


    y_pred: [N,1]      y_true: [N], [N,1]
        binary_cross_entropy