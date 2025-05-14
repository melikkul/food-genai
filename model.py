import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess recipe data
with open('tarifler.json') as f:
    recipes = json.load(f)

# Process ingredients: remove quantities (text in parentheses) and combine
ingredient_texts = []
for recipe in recipes:
    ingredients = ' '.join([ing.split('(')[0].strip() for ing in recipe['malzemeler']])
    ingredient_texts.append(ingredients)

# Convert text to numerical sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ingredient_texts)
total_words = len(tokenizer.word_index) + 1  # Add 1 for out-of-vocabulary words

# Create training sequences using sliding window approach
input_sequences = []
for text in ingredient_texts:
    token_list = tokenizer.texts_to_sequences([text])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Standardize sequence length for model input
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Prepare training data (features and labels)
X = input_sequences[:, :-1]  # All elements except last as input
y = input_sequences[:, -1]   # Last element as prediction target
y = to_categorical(y, num_classes=total_words)  # Convert to one-hot encoding

# Neural network architecture for sequence prediction
model = Sequential([
    # Convert word indices to dense vectors
    Embedding(total_words, 64, input_length=max_sequence_len-1),
    # First LSTM layer with dropout to prevent overfitting
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    # Second LSTM layer for final sequence processing
    LSTM(64),
    # Dense layers for classification
    Dense(64, activation='relu'),
    Dense(total_words, activation='softmax')  # Output probabilities for all words
])

# Configure model for training
model.compile(loss='categorical_crossentropy', 
             optimizer='adam',  # Effective default optimizer
             metrics=['accuracy'])

model.summary()

# Train model and save weights
history = model.fit(X, y, epochs=1000, verbose=1)
model.save("tarifler.h5")

def generate_recipe(seed_text, next_words=10, temperature=0.7):
    """
    Generates new ingredient list using trained model
    seed_text: Starting words for generation
    next_words: Number of words to generate
    temperature: Controls randomness (0.0-1.0)
    """
    for _ in range(next_words):
        # Convert current text to numerical sequence
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # Match model's expected input format
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        
        # Get model's prediction with temperature sampling
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
        
        # Convert predicted index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage: Generate recipe starting with "Tavuk" (Chicken)
print("Üretilen Tarif Malzemeleri:")
print(generate_recipe("Tavuk", next_words=15, temperature=0.6))