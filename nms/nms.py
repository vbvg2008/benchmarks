import tensorflow as tf

cls_pred = tf.random.uniform([32, 100, 5])
loc_pred = tf.random.uniform([32, 100, 4])

top_n = 50
score_threshold = 0.05
num_anchor = loc_pred.shape[1]
cls_best_score = tf.reduce_max(cls_pred, axis=-1)
cls_best_class = tf.argmax(cls_pred, axis=-1)

#select top n anchor boxes to proceed 
sorted_score = tf.sort(cls_best_score,  direction='DESCENDING')
cls_best_score = tf.cond(tf.greater(num_anchor, 50),
                         lambda: tf.where(tf.greater_equal(cls_best_score, tf.tile(sorted_score[:,top_n-1:top_n],[1, num_anchor])), cls_best_score, 0.0),
                         lambda: cls_best_score)

#Padded Nonmax suppression with threshold
selected_indices_padded = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True, score_threshold=score_threshold).selected_indices, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)
valid_outputs = tf.map_fn(lambda x: tf.image.non_max_suppression_padded(x[0], x[1], top_n, pad_to_max_output_size=True, score_threshold=score_threshold).valid_outputs, (loc_pred, cls_best_score), dtype=tf.int32, back_prop=False)