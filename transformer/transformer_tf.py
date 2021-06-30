import pdb

import fastestimator as fe
import numpy as np
import tensorflow as tf
from fastestimator.dataset.data import tednmt
from tensorflow.keras import layers
from transformers import BertTokenizer


class Encode(fe.op.numpyop.NumpyOp):
    def __init__(self, tokenizer, inputs, outputs, mode=None):
        super().__init__(inputs=inputs, outputs=outputs, mode=mode)
        self.tokenizer = tokenizer

    def forward(self, data, state):
        return np.array(self.tokenizer.encode(data))


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  # this is to make the softmax of masked cells to be 0
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output


def point_wise_feed_forward_network(em_dim, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(em_dim)  # (batch_size, seq_len, em_dim)
    ])


class MultiHeadAttention(layers.Layer):
    def __init__(self, em_dim, num_heads):
        super().__init__()
        assert em_dim % num_heads == 0, "model dimension must be multiply of number of heads"
        self.num_heads = num_heads
        self.em_dim = em_dim
        self.depth = em_dim // self.num_heads
        self.wq = layers.Dense(em_dim)
        self.wk = layers.Dense(em_dim)
        self.wv = layers.Dense(em_dim)
        self.dense = layers.Dense(em_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # B, num_heads, seq_len, depth

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  # B, seq_len, em_dim
        k = self.wk(k)  # B, seq_len, em_dim
        v = self.wv(v)  # B, seq_len, em_dim
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  #B, seq_len, num_heads, depth
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.em_dim))  # B, seq_len, em_dim
        output = self.dense(concat_attention)
        return output


class EncoderLayer(layers.Layer):
    def __init__(self, em_dim, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(em_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(em_dim, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


def get_angles(pos, i, em_dim):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(em_dim))
    return pos * angle_rates


def positional_encoding(position, em_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(em_dim)[np.newaxis, :], em_dim)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


class DecoderLayer(layers.Layer):
    def __init__(self, em_dim, num_heads, diff, rate=0.1):
        super().__init__()
        self.mha1 = MultiHeadAttention(em_dim, num_heads)
        self.mha2 = MultiHeadAttention(em_dim, num_heads)
        self.ffn = point_wise_feed_forward_network(em_dim, diff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_out, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2 = self.mha2(enc_out, enc_out, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3


class Encoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, input_vocab, max_pos_enc, rate=0.1):
        super().__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.embedding = layers.Embedding(input_vocab, em_dim)
        self.pos_encoding = positional_encoding(max_pos_enc, self.em_dim)
        self.enc_layers = [EncoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.em_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, em_dim, num_heads, dff, target_vocab, max_pos_enc, rate=0.1):
        super().__init__()
        self.em_dim = em_dim
        self.num_layers = num_layers
        self.embedding = layers.Embedding(target_vocab, em_dim)
        self.pos_encoding = positional_encoding(max_pos_enc, em_dim)
        self.dec_layers = [DecoderLayer(em_dim, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=None):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.em_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
        return x


def transformer(num_layers, em_dim, num_heads, dff, input_vocab, target_vocab, max_pos_enc, max_pos_dec, rate=0.1):
    inputs = layers.Input(shape=(None, ))
    targets = layers.Input(shape=(None, ))
    enc_padding_mask = layers.Input(shape=(None, ))
    look_ahead_mask = layers.Input(shape=(None, ))
    dec_padding_mask = layers.Input(shape=(None, ))
    x = Encoder(num_layers, em_dim, num_heads, dff, input_vocab, max_pos_enc, rate=rate)(inputs, enc_padding_mask)
    x = Decoder(num_layers, em_dim, num_heads, dff, target_vocab, max_pos_dec, rate=rate)(targets,
                                                                                          x,
                                                                                          look_ahead_mask,
                                                                                          dec_padding_mask)
    x = layers.Dense(target_vocab)(x)
    model = tf.keras.Model(inputs=[inputs, targets, enc_padding_mask, look_ahead_mask, dec_padding_mask], outputs=x)
    return model


def fastestimator_run():
    train_ds, eval_ds, test_ds = tednmt.load_data(translate_option="pt_to_en")
    pt_tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
    en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    pipeline = fe.Pipeline(
        train_data=train_ds,
        eval_data=eval_ds,
        test_data=test_ds,
        batch_size=64,
        ops=[
            Encode(inputs="source", outputs="source", tokenizer=pt_tokenizer),
            Encode(inputs="target", outputs="target", tokenizer=en_tokenizer)
        ],
        pad_value=0)
    pipeline.benchmark()


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


if __name__ == "__main__":
    model = transformer(num_layers=2,
                        em_dim=512,
                        num_heads=8,
                        dff=2048,
                        input_vocab=8500,
                        target_vocab=8000,
                        max_pos_enc=10000,
                        max_pos_dec=6000)
    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(temp_input, temp_target)
    out = model([temp_input, temp_target, enc_padding_mask, combined_mask, dec_padding_mask], training=False)
    # pdb.set_trace()
