# Telugu BPE Tokenizer

This is a Byte Pair Encoding (BPE) tokenizer specifically designed for Telugu text processing. The model has been trained on Telugu text data with a vocabulary size of approximately 5000 tokens.

## Features

- Telugu-specific text preprocessing
- BPE tokenization with ~5000 tokens vocabulary
- Handles Telugu characters, numbers, and punctuation
- Provides compression statistics

## Compression Performance

The tokenizer achieves efficient text compression with the following characteristics:
- Average compression ratio: ~1.5x - 2.5x (depending on input text)
- Vocabulary size: ~5000 tokens
- Handles both simple and complex Telugu text structures
- Preserves linguistic features while reducing token count

Example compression results:

## Usage

Enter Telugu text in the input box and get:
- Preprocessed text
- Tokenized output
- Character and token counts
- Compression ratio

## Examples

Try these sample inputs:
- నమస్కారం (Hello)
- తెలుగు భాష చాలా అందమైన భాష (Telugu is a beautiful language)
- నేను తెలుగులో మాట్లాడగలను (I can speak in Telugu)

## Model Details

- Vocabulary Size: ~5000 tokens
- Preprocessing: Telugu-specific regex patterns
- Implementation: Python-based BPE algorithm 