import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataIngestor:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def extract_texts_and_labels(self, dataset):
        """
        Extracts text and labels from the dataset and returns a list of strings.
        """
        texts, labels = [], []
        for text, label in dataset:
            texts.append(text.numpy().decode("utf-8"))
            labels.append(label.numpy().decode("utf-8"))
        return texts + labels  # Combine texts and labels for tokenization
    
    def text_to_seq(self, text, label):
        """
        Converts text and label to numerical sequences using tokenizer.
        """
        text_seq = self.tokenizer.texts_to_sequences([text.numpy().decode("utf-8")])[0]
        label_seq = self.tokenizer.texts_to_sequences([label.numpy().decode("utf-8")])[0]
        return tf.constant(text_seq, dtype=tf.int32), tf.constant(label_seq, dtype=tf.int32)

    def pad_sequence(self, text, label):
        """
        Pads text and label sequences to a fixed length.
        """
        text_padded = pad_sequences([text.numpy()], maxlen=32, padding='post')[0]
        label_padded = pad_sequences([label.numpy()], maxlen=128, padding='post')[0]
        return tf.constant(text_padded, dtype=tf.int32), tf.constant(label_padded, dtype=tf.int32)

    def apply_pad_seq(self, text, label):
        """
        Wrapper to apply padding using tf.py_function.
        """
        text, label = tf.py_function(self.pad_sequence, [text, label], [tf.int32, tf.int32])
        text.set_shape([32])
        label.set_shape([128])
        return text, label


def prepare_datasets():
    """
    Loads dataset, processes text, tokenizes, and prepares the TensorFlow dataset pipeline.
    """
    data_ingestor = DataIngestor()
    
    # Load dataset
    ds = tfds.load("piqa")
    train_ds, valid_ds = ds["train"], ds["validation"]
    
    # Extract text and labels
    train_ds = train_ds.map(lambda sample: (sample["goal"], sample["sol1"]))
    valid_ds = valid_ds.map(lambda sample: (sample["goal"], sample["sol1"]))

    # Collect data for tokenizer training
    data_to_tokenize = data_ingestor.extract_texts_and_labels(train_ds) + data_ingestor.extract_texts_and_labels(valid_ds)
    
    # Train tokenizer
    data_ingestor.tokenizer.fit_on_texts(data_to_tokenize)

    # Apply text-to-sequence conversion
    train_ds = train_ds.map(lambda text, label: tf.py_function(data_ingestor.text_to_seq, [text, label], [tf.int32, tf.int32]))
    valid_ds = valid_ds.map(lambda text, label: tf.py_function(data_ingestor.text_to_seq, [text, label], [tf.int32, tf.int32]))

    # Apply padding
    train_ds = train_ds.map(data_ingestor.apply_pad_seq)
    valid_ds = valid_ds.map(data_ingestor.apply_pad_seq)

    # Batch and optimize pipeline
    batch_size = 32
    train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, valid_ds


# Expose datasets for other files
train_ds, valid_ds = prepare_datasets()