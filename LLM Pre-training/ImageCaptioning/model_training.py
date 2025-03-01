import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, LSTM, Embedding, Dense, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preparation import X1, X2, y, data_ingestor, features_dict, resnet_model, dataset_path

class ImageCaptioning:
    """
    Defines the encoder-decoder architecture for image captioning.
    """

    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the encoder-decoder model for caption generation.
        """
        # Image feature input
        image_input = Input(shape=(2048,))
        fe1 = Dropout(0.5)(image_input)
        fe2 = Dense(256, activation='relu')(fe1)

        # Text input
        text_input = Input(shape=(data_ingestor.max_length,))
        se1 = Embedding(data_ingestor.vocab_size, 256, mask_zero=True)(text_input)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        # Merge image and text features
        decoder1 = Add()([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)

        # Final output: probability distribution over the vocabulary
        outputs = Dense(data_ingestor.vocab_size, activation='softmax')(decoder2)

        # Define the model
        model = Model(inputs=[image_input, text_input], outputs=outputs)
        return model
    
    def generate_caption(self, image_feature):
        """
        Generates captions for an image using greedy search.
        """
        in_text = 'startseq'

        for i in range(data_ingestor.max_length):
            sequence = data_ingestor.tokenizer.text_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=data_ingestor.max_length, padding='post')

            yhat = self.model.predict([np.array(image_feature), sequence], verbose=0)
            yhat = np.argmax(yhat)

            word = None
            for w, index in data_ingestor.tokenizer.word_index.items():
                if yhat == index:
                    word = w
                    break

            if word is None:
                break
            in_text += ' ' + word
            if word == 'endseq':
                break
            
        return in_text.replace('startseq', '').replace('endseq', '').strip()



if __name__ == "__main__":
    captioning_model = ImageCaptioning()
    captioning_model.model.compile(loss='categorical_crossentropy', optimizer='adam')
    captioning_model.model.summary()

    # Train model
    epochs = 10
    batch_size = 64
    history = captioning_model.model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Save model
    captioning_model.model.save('image_captioning_model.h5')

    # ----------------------------
    # Generate Caption for an Image
    # ----------------------------
    sample_image_id = list(features_dict.keys())[0]
    sample_image_path = os.path.join(dataset_path, 'Images', sample_image_id)
    sample_feature = data_ingestor.extract_features(sample_image_path, resnet_model)
    caption = captioning_model.generate_caption(sample_feature)
    print('Generated Caption: ', caption )