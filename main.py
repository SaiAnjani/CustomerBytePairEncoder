import re
import requests
from collections import Counter, defaultdict
import pandas as pd
from collections import defaultdict

def get_vocab_1(sentences, vocab_size=1000):
  """
  Builds a vocabulary from a list of sentences.

  Args:
    sentences: A list of sentences, where each sentence is a list of words.
    vocab_size: The desired size of the vocabulary.

  Returns:
    A dictionary mapping words to their integer indices.
  """
  word_counts = {}
  print()
  for word in sentences.split():
     word_counts[word] = word_counts.get(word, 0) + 1
#   for sentence in sentences:


  vocab = sorted(word_counts, key=word_counts.get, reverse=True)[:vocab_size-1]
  
#   vocab.append('<UNK>')  # Add unknown token
  vocab_dict = {word: i for i, word in enumerate(vocab)}
  print(vocab_dict)
#   print()
  return vocab_dict

def bpe_train(sentences, vocab_size, num_merges=10000):
  """
  Trains a Byte Pair Encoding (BPE) model.

  Args:
    sentences: A list of sentences, where each sentence is a list of words.
    vocab_size: The desired size of the vocabulary.
    num_merges: The number of merge operations to perform.

  Returns:
    A list of merge operations, where each operation is a tuple of two subwords.
  """
  vocab = get_vocab_1(sentences, vocab_size)
  word_counts = {}
  for word in sentences.split():
    word_counts[word] = word_counts.get(word, 0) + 1
#   for sentence in sentences:


  bigrams = defaultdict(int)
  print('hi')
  print(word_counts)
  for word, count in word_counts.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      bigram = (symbols[i], symbols[i+1])
      bigrams[bigram] += count

  merges = []
  for _ in range(num_merges):
    if not bigrams:
      break
    max_bigram = max(bigrams, key=bigrams.get)
    merges.append(max_bigram)

    # Merge the most frequent bigram
    bigram_str = ''.join(max_bigram)
    for bigram, count in list(bigrams.items()):
      if bigram[0] == max_bigram[0] and bigram[1] == max_bigram[1]:
        bigrams[(''.join(bigram),)] = count
        del bigrams[bigram]

      elif bigram[0] == max_bigram[0] and bigram[1] != max_bigram[1]:
        new_bigram = (bigram_str, bigram[1])
        bigrams[new_bigram] += count
        del bigrams[bigram]

      elif bigram[0] != max_bigram[0] and bigram[1] == max_bigram[1]:
        new_bigram = (bigram[0], bigram_str)
        bigrams[new_bigram] += count
        del bigrams[bigram]

  return merges

def bpe_encode(word, merges):
  """
  Encodes a word using a given list of merge operations.

  Args:
    word: The word to encode.
    merges: A list of merge operations.

  Returns:
    A list of subwords.
  """
  symbols = word.split()
  for bigram in reversed(merges):
    bigram_str = ''.join(bigram)
    while bigram_str in ''.join(symbols):
      symbols = symbols.replace(bigram_str, bigram)
  return symbols

def calculate_compression_ratio(original_text, encoded_text):
  """
  Calculates the compression ratio.

  Args:
    original_text: The original text.
    encoded_text: The encoded text.

  Returns:
    The compression ratio.
  """
  original_size = len(original_text)
  encoded_size = len(encoded_text)
  return original_size / encoded_size

# Example usage


# Step 1: Download telugu Dataset
def download_telugu_dataset(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

# Step 2: Pre-process telugu Text
def preprocess_telugu_text(text):
    # Remove unwanted characters and tokenize
    text = re.sub(r'[^\u0C00-\u0C7F\s]', '', text)  # Keep only Telugu characters and spaces
  # Keep only telugu characters and spaces
    words = text.split()
    return words

# Step 3: Implement BPE
def get_vocab(words):
    vocab = Counter()
    for word in words:
        word = ' '.join(list(word)) + ' </w>'
        vocab[word] += 1
    return vocab

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in vocab:
        w_out = p.sub(''.join(pair), word)
        new_vocab[w_out] = vocab[word]
    return new_vocab

def byte_pair_encoding(words, vocab_size):
    vocab = get_vocab(words)
    for i in range(vocab_size - len(vocab)):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    return vocab

# Step 4: Generate Vocabulary
def generate_vocabulary(text, vocab_size=5000):
    words = preprocess_telugu_text(text)
    print(words[500:600])
    vocab = byte_pair_encoding(words, vocab_size)
    return vocab




# Step 5: Ensure Compression
def calculate_compression(original_text, vocab):
    original_size = len(original_text)
    compressed_size = sum(len(word) for word in vocab)
    print(f'Original Size: {original_size} and Compressed Size: {compressed_size}')
    compression_ratio = original_size / compressed_size
    return compression_ratio

import json

def save_vocab(vocab, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

def save_model(vocab, model_dir):
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_vocab(vocab, os.path.join(model_dir, 'vocab.json'))


def read_text_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text
# Main function
def main():
    # temp_df = pd.read_csv('telugu_books.csv', encoding='utf-8', quotechar='"',on_bad_lines='skip',skiprows=[20295,23622,25228])
    # temp_df = temp_df[~temp_df['text'].isna()]
    # result = temp_df[1:1000]['text'].str.cat(sep=' ')
    # telugu_text = result[1:75000]

    filepath = '14529.txt'
    telugu_text = read_text_from_file(filepath)
    print(len(telugu_text))

    # vocab = generate_vocabulary(telugu_text)

#     sentences = [
#     "this is the first sentence".split(),
#     "this is the second sentence".split(),
#     "and this is the third one".split()
# ]
    vocab_size = 5000
    num_merges = 100

    merges = bpe_train(telugu_text, vocab_size, num_merges)
    print("Merge operations:", merges)

    word = "sentence"
    encoded_word = bpe_encode(word, merges)
    print("Encoded word:", encoded_word)

    # original_text = " ".join(sentences)
    encoded_text = " ".join([" ".join(bpe_encode(word, merges)) for word in telugu_text.split()]) 

    compression_ratio = calculate_compression_ratio(telugu_text, encoded_text)
    print("Compression Ratio:", compression_ratio)




    # compression_ratio = calculate_compression(telugu_text, vocab)
    
    # print(f"Vocabulary Size: {len(vocab)}")
    # print(f"Compression Ratio: {compression_ratio}")
    # if compression_ratio > 3.2:
    #     print("Compression size is more than 3.2")
    # else:
    #     print("Compression size is less than 3.2")

    # save_model(vocab, 'telugu_bpe_model')


if __name__ == "__main__":
    main()
