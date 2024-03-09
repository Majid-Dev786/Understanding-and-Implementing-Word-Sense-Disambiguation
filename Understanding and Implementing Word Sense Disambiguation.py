# Importing necessary libraries and modules
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Input sentences and corresponding labels
sentences = [
    "I saw a bear at the zoo.",
    "He went to bear the weight of the world.",
    "The tree had a lot of bear fruits."
]

labels = [0, 1, 0]  

class TextProcessor:
    # Constructor for TextProcessor class
    def __init__(self, sentences, labels):
        # Initializing TextProcessor object with input sentences and labels
        self.sentences = sentences
        self.labels = tf.convert_to_tensor(labels, dtype=tf.float32)  
        self.tokenizer = Tokenizer()

    # Tokenize input sentences and pad the sequences
    def tokenize_sentences(self):
        self.tokenizer.fit_on_texts(self.sentences)
        self.word_index = self.tokenizer.word_index
        sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.max_length = max([len(seq) for seq in sequences])
        return pad_sequences(sequences, maxlen=self.max_length, padding='post')

class AnimalSenseModel:
    # Constructor for AnimalSenseModel class
    def __init__(self, word_index, max_length, tokenizer):
        # Initializing AnimalSenseModel object with word_index, max_length, and tokenizer
        self.word_index = word_index
        self.max_length = max_length
        self.model = self.build_model()
        self.tokenizer = tokenizer 

    # Build and compile the model
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(len(self.word_index) + 1, 16, input_length=self.max_length),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # Train the model
    def train_model(self, padded_sequences, labels, epochs=10):
        self.model.fit(padded_sequences, labels, epochs=epochs)

    # Predict on new sentences
    def predict_sentences(self, test_sentences):
        test_sequences = self.tokenizer.texts_to_sequences(test_sentences)  
        padded_test_sequences = pad_sequences(test_sequences, maxlen=self.max_length, padding='post')
        return self.model.predict(padded_test_sequences)

# Print predictions for test sentences
def print_predictions(test_sentences, predictions):
    for i, sentence in enumerate(test_sentences):
        print(f"Sentence: {sentence}")
        print(f"Prediction: {predictions[i][0]} (0 represents non-animal sense, 1 represents animal sense)")

if __name__ == "__main__":
    # Create instance of TextProcessor and tokenize input sentences
    text_processor = TextProcessor(sentences, labels)
    padded_sequences = text_processor.tokenize_sentences()

    # Create instance of AnimalSenseModel and train the model
    animal_sense_model = AnimalSenseModel(text_processor.word_index, text_processor.max_length, text_processor.tokenizer)
    animal_sense_model.train_model(padded_sequences, text_processor.labels)

    # Test sentences for prediction
    test_sentences = [
        "I can't bear this pain anymore.",
        "She decided to bear the burden."
    ]
    
    # Predict and print results
    predictions = animal_sense_model.predict_sentences(test_sentences)
    print_predictions(test_sentences, predictions)
