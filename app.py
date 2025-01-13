import gradio as gr
from telugu_bpe import TeluguBPE
import os

# Initialize the BPE model
bpe = TeluguBPE(vocab_size=5000)

# Get the absolute path to the model file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "telugu_bpe_model.json")

# Load the pre-trained model
try:
    bpe.load_model(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    # Train a small model with sample text if model doesn't exist
    sample_text = """
    నమస్కారం తెలుగు భాష చాలా అందమైన భాష
    తెలుగు భారతదేశంలోని ద్రావిడ భాషల్లో ఒకటి
    తెలుగు అక్షరమాల లో 56 అక్షరాలు ఉన్నాయి
    """
    processed_text = bpe.preprocess_telugu_text(sample_text)
    bpe.learn_bpe(processed_text)
    bpe.save_model(model_path)
    print("Created a new model with sample text")

def process_text(input_text: str) -> dict:
    """
    Process input Telugu text and return tokenization results
    """
    if not input_text or input_text.strip() == "":
        return {
            "Error": "Please enter some Telugu text"
        }
    
    try:
        # Preprocess the input text
        processed_text = bpe.preprocess_telugu_text(input_text)
        
        # Encode the text
        encoded_tokens = bpe.encode(processed_text)
        
        # Calculate statistics
        char_count = len(processed_text)
        token_count = len(encoded_tokens)
        compression_ratio = char_count / token_count if token_count > 0 else 0
        
        return {
            "Preprocessed Text": processed_text,
            "Tokens": encoded_tokens,
            "Character Count": char_count,
            "Token Count": token_count,
            "Compression Ratio": f"{compression_ratio:.2f}x",
            "Vocabulary Size": len(bpe.vocab)
        }
    except Exception as e:
        return {
            "Error": f"An error occurred: {str(e)}"
        }

# Create Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            lines=4, 
            placeholder="Enter Telugu text here...",
            label="Input Telugu Text",
            value="నమస్కారం"
        )
    ],
    outputs=gr.JSON(label="Tokenization Results"),
    title="Telugu BPE Tokenizer",
    description="""
    ## Telugu Byte Pair Encoding (BPE) Tokenizer
    
    This tokenizer is specifically designed for Telugu text processing with a vocabulary size of ~5000 tokens.
    
    ### Features:
    - Telugu-specific preprocessing
    - BPE tokenization
    - Compression statistics
    - Character and token counts
    
    ### How to use:
    1. Enter Telugu text in the input box
    2. Get tokenized output and statistics
    
    ### Example inputs provided below ⬇️
    """,
    examples=[
        ["నమస్కారం"],
        ["తెలుగు భాష చాలా అందమైన భాష"],
        ["నేను తెలుగులో మాట్లాడగలను"],
        ["తెలుగు అక్షరమాల లో 56 అక్షరాలు ఉన్నాయి"]
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
    cache_examples=True
)

# Launch configuration for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    ) 