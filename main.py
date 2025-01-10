import re
import requests
from collections import Counter, defaultdict

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
def generate_vocabulary(text, vocab_size=30000):
    words = preprocess_telugu_text(text)
    print(words[500:600])
    vocab = byte_pair_encoding(words, vocab_size)
    return vocab

# Step 5: Ensure Compression
def calculate_compression(original_text, vocab):
    original_size = len(original_text)
    compressed_size = sum(len(word) for word in vocab)
    compression_ratio = original_size / compressed_size
    return compression_ratio

# Main function
def main():
    temp_df = pd.read_csv('telugu_books.csv', encoding='utf-8', quotechar='"',on_bad_lines='skip',skiprows=[20295,23622,25228])
    temp_df = temp_df[~temp_df['text'].isna()]
    result = temp_df[1:1000]['text'].str.cat(sep=' ')
    telugu_text = result[1:75000]
    vocab = generate_vocabulary(telugu_text)
    compression_ratio = calculate_compression(telugu_text, vocab)
    
    print(f"Vocabulary Size: {len(vocab)}")
    print(f"Compression Ratio: {compression_ratio}")
    if compression_ratio > 3.2:
        print("Compression size is more than 3.2")
    else:
        print("Compression size is less than 3.2")

    print(vocab)

if __name__ == "__main__":
    main()
