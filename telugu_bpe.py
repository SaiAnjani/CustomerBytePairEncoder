import re
import collections
from typing import Dict, List, Tuple, Set
import json
from pathlib import Path

class TeluguBPE:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Set[str] = set()
        
    def preprocess_telugu_text(self, text: str) -> str:
        """
        Preprocess Telugu text with specific rules
        """
        # Remove any ASCII characters except spaces and newlines
        text = re.sub(r'[^\u0C00-\u0C7F\s\n]', '', text)
        
        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Add spaces between Telugu characters and numbers
        text = re.sub(r'(\d+)', r' \1 ', text)
        
        # Add spaces between Telugu punctuation marks
        text = re.sub(r'([।॥,?!])', r' \1 ', text)
        
        # Handle Telugu specific patterns
        # Add space after purna virama (full stop)
        text = re.sub(r'([।॥])', r'\1 ', text)
        
        # Separate combined vowel marks
        text = re.sub(r'([\u0C3E-\u0C4C])', r' \1', text)
        
        return text.strip()

    def get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """
        Count frequency of adjacent pairs in current vocabulary
        """
        pairs = collections.defaultdict(int)
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i + 2])] += 1
        return pairs

    def merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """
        Merge all occurrences of the most frequent pair
        """
        first, second = pair
        new_words = []
        
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
            
        return new_words

    def learn_bpe(self, text: str) -> None:
        """
        Learn BPE merges from text
        """
        # Initial vocabulary: character level
        words = [[char for char in word] for word in text.split()]
        self.vocab = set(char for word in words for char in word)
        
        num_merges = self.vocab_size - len(self.vocab)
        
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break
                
            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = best_pair[0] + best_pair[1]
            self.vocab.add(self.merges[best_pair])
            
            words = self.merge_vocab(words, best_pair)
            
            if len(self.vocab) >= self.vocab_size:
                break

    def encode(self, text: str) -> List[str]:
        """
        Encode text using learned BPE merges
        """
        words = [[char for char in word] for word in text.split()]
        for pair, merge in self.merges.items():
            words = self.merge_vocab(words, pair)
        return [token for word in words for token in word]

    def save_model(self, path: str) -> None:
        """
        Save BPE model to file
        """
        model_data = {
            'vocab_size': self.vocab_size,
            'merges': {f'{k[0]} {k[1]}': v for k, v in self.merges.items()},
            'vocab': list(self.vocab)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

    def load_model(self, path: str) -> None:
        """
        Load BPE model from file
        """
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.vocab_size = model_data['vocab_size']
        self.merges = {tuple(k.split()): v for k, v in model_data['merges'].items()}
        self.vocab = set(model_data['vocab'])

def main():
    # Example usage
    input_file = "telugu_text.txt"
    model_file = "telugu_bpe_model.json"
    
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f'Started learning BPE')
    bpe = TeluguBPE(vocab_size=5000)
    
    # Preprocess text
    processed_text = bpe.preprocess_telugu_text(text)
    
    # Calculate original text statistics
    original_chars = len(processed_text)
    original_tokens = len(processed_text.split())
    
    # Learn BPE
    bpe.learn_bpe(processed_text)
    
    # Encode the entire text to calculate compression
    encoded_text = bpe.encode(processed_text)
    encoded_length = len(encoded_text)
    
    # Calculate compression ratio
    compression_ratio = original_chars / encoded_length
    
    # Save model
    bpe.save_model(model_file)
    
    # Print statistics
    print(f"\nCompression Statistics:")
    print(f"Original characters: {original_chars}")
    print(f"Original tokens (words): {original_tokens}")
    print(f"Encoded tokens: {encoded_length}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Vocabulary size: {len(bpe.vocab)}")
    
    # Example encoding
    sample_text = "నమస్కారం"  # "Hello" in Telugu
    encoded = bpe.encode(bpe.preprocess_telugu_text(sample_text))
    print(f"\nExample encoding:")
    print(f"Sample text: {sample_text}")
    print(f"Encoded text: {encoded}")

if __name__ == "__main__":
    main() 