import tensorflow as tf

cls_pred = tf.random.uniform([32, 100, 5])
loc_pred = tf.random.uniform([32, 100, 4])

cls_best_score = tf.reduce_max(cls_pred, axis=-1)
cls_best_class = tf.argmax(cls_pred, axis=-1)

#here needed to get the top 1k scores
sorted_score = tf.sort(cls_best_score,  direction='DESCENDING')
tf.cond()

selected_indices_padded = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], 100, pad_to_max_output_size=True, score_threshold=0.05).selected_indices, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
valid_outputs = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], 100, pad_to_max_output_size=True, score_threshold=0.05).valid_outputs, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)