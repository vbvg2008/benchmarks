import tensorflow as tf

cls_pred = tf.random.uniform([32, 100, 5])
loc_pred = tf.random.uniform([32, 100, 4])

cls_best_score = tf.reduce_max(cls_pred, axis=-1)
cls_best_class = tf.argmax(cls_pred, axis=-1)
select_idx = tf.where(tf.greater(cls_best_score, 0.5))


selected_indices_padded = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], 100, pad_to_max_output_size=True).selected_indices, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False, infer_shape=False)
valid_outputs = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], 100, pad_to_max_output_size=True).valid_outputs, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False, infer_shape=False)




num_valid = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], 100, pad_to_max_output_size=True).selected_indices, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False, infer_shape=False)


idx0 = tf.image.non_max_suppression(loc_pred[0], cls_best_score[0], 100)
idx1 = tf.image.non_max_suppression(loc_pred[1], cls_best_score[1], 100)
a  = tf.image.non_max_suppression_padded(loc_pred[0], cls_best_score[0], 100, pad_to_max_output_size=True)
selected_indices_padded, num_valid  = tf.image.non_max_suppression_padded(loc_pred[1], cls_best_score[1], 100, pad_to_max_output_size=True)