import os
import time
import torch
import pickle
import argparse
from contextlib import nullcontext
import tiktoken  # Import tiktoken

# Assuming 'model.py' defines the Transformer correctly and its generate method
# handles temperature, top_k, and tiktoken_vocab_size as implemented previously.
import model
from model import Transformer

parser = argparse.ArgumentParser(description="Generate text using a Transformer model with Tiktoken.")

parser.add_argument('--ckpath', type=str, required=True, help='Path to the model checkpoint (.pt file)')
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing meta.pkl (must contain tiktoken info)')
parser.add_argument('--prompt', type=str, default="Hello!", help='Starting prompt for generation')
# Model architecture arguments (should match the loaded checkpoint)
# IMPORTANT: These should match the *padded* configuration if padding was used during training
parser.add_argument('--n_embd', type=int, default=768, help='Embedding dimension (match checkpoint)')
parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads (match checkpoint)')
parser.add_argument('--n_layer', type=int, default=12, help='Number of layers (match checkpoint)')
parser.add_argument('--n_experts', type=int, default=None, help='Number of experts per MoE layer (if used, match checkpoint)')
parser.add_argument('--ctx_len', type=int, default=1024, help='Context length (match checkpoint training or set for generation)')
# Generation arguments
parser.add_argument('--max_tok', type=int, default=100, help='Maximum number of new tokens to generate')
parser.add_argument('--temp', type=float, default=0.1, help='Sampling temperature (e.g., 0.8 for less random, 1.0 for standard, >1.0 for more random)')
parser.add_argument('--top_k', type=int, default=5000, help='Top-k sampling threshold (e.g., 50). Use 0 or None to disable.')
# Optional: Specify model types if needed by your model architecture
parser.add_argument('--types', nargs='*', type=str, default=['mlp'], help='Types of layers used (e.g., mlp moe) - match checkpoint')
# Optional: Add other config args if your model.py needs them (like init_moe_scaling)
parser.add_argument('--init_moe_scaling', type=float, default=1.0, help='Initial MoE scaling factor (match checkpoint if applicable)') # Example

args = parser.parse_args()

# --- Configuration and Device Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    # Basic check, specific GPU name might differ
    try:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Could not get GPU name: {e}")


use_compile = hasattr(torch, 'compile') and device=='cuda'

# --- Load Tokenizer Info from Meta ---
meta_path = os.path.join(args.data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    print(f"Error: meta.pkl not found at {meta_path}")
    exit(1)

print(f"Loading metadata from {meta_path}...")
enc = None # Initialize enc to None
model_vocab_size = None # Vocab size for the model architecture
tiktoken_vocab_size = None # Vocab size reported by tiktoken

try:
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    if 'vocab_size' not in meta or 'tokenizer' not in meta:
        print("Error: meta.pkl must contain 'vocab_size' and 'tokenizer' (tiktoken model name)")
        exit(1)

    # This is the vocab size the model was *trained* with (potentially padded)
    model_vocab_size = meta['vocab_size']
    tiktoken_model_name = meta['tokenizer']
    print(f"  Model vocab size (from meta.pkl): {model_vocab_size:,}")
    print(f"  Tokenizer model name: {tiktoken_model_name}")

except Exception as e:
    print(f"Error loading or parsing meta.pkl: {e}")
    exit(1)

# --- Initialize Tiktoken Encoder ---
try:
    enc = tiktoken.get_encoding(tiktoken_model_name)
    encode = lambda s: enc.encode(s, allowed_special='all')
    decode = lambda l: enc.decode(l)
    tiktoken_vocab_size = enc.n_vocab # Get the actual size from tiktoken
    print(f"Tiktoken encoder '{tiktoken_model_name}' loaded successfully.")
    print(f"  Tiktoken vocabulary size: {tiktoken_vocab_size:,}")

    # Check for mismatch and warn if necessary
    if model_vocab_size != tiktoken_vocab_size:
         print(f"\n!!! Vocab Size Mismatch Detected !!!")
         print(f"  Model expects vocab size: {model_vocab_size} (from meta.pkl)")
         print(f"  Tiktoken provides vocab size: {tiktoken_vocab_size}")
         if model_vocab_size > tiktoken_vocab_size:
             padding = model_vocab_size - tiktoken_vocab_size
             print(f"  This suggests the model was trained with a vocabulary padded by {padding} tokens.")
             print(f"  Generation will proceed, ignoring the padded token indices (>= {tiktoken_vocab_size}).")
         else:
             print(f"  Warning: Model vocab size is smaller than Tiktoken's. This is unusual.")
             print(f"  Proceeding, but the model might not be able to generate all tokens Tiktoken knows.")
    else:
        print("  Model and Tiktoken vocabulary sizes match.")

except Exception as e:
    print(f"Error initializing tiktoken encoder '{tiktoken_model_name}': {e}")
    exit(1)

# --- Configure and Load Model ---
print("\nConfiguring model...")
# Use the vocabulary size from meta.pkl for model configuration
model.config['device'] = device
model.config['vocab_size'] = model_vocab_size # CRITICAL: Use the potentially padded size
model.config['ctx_len'] = args.ctx_len
model.config['n_embd'] = args.n_embd
model.config['n_head'] = args.n_head
model.config['n_layer'] = args.n_layer
model.config['n_experts'] = args.n_experts
model.config['type'] = args.types
# Add any other relevant config args from command line or defaults
model.config['init_moe_scaling'] = args.init_moe_scaling # Example
# Make sure dropout is set for eval mode (might be handled in model.eval() but good to be explicit if needed)
# model.config['dropout'] = 0.0 # Typically dropout is disabled during eval

print("Model Configuration:")
# Filter config to show relevant items, avoid printing huge tensors if any
relevant_keys = ['device', 'vocab_size', 'ctx_len', 'n_embd', 'n_head', 'n_layer', 'n_experts', 'type', 'init_moe_scaling', 'dropout']
for key in relevant_keys:
    if key in model.config:
        print(f"  {key}: {model.config[key]}")

transformer_model = Transformer()

print(f"\nLoading model checkpoint from {args.ckpath}...")
try:
    # --- MODIFICATION: Use weights_only=False as discussed ---
    print("Warning: Loading checkpoint with weights_only=False. Ensure the checkpoint source is trusted.")
    checkpoint = torch.load(args.ckpath, map_location=device, weights_only=False)
    # --- END MODIFICATION ---

    # --- Add Check: Ensure checkpoint is a dictionary containing 'model' ---
    if not isinstance(checkpoint, dict):
        print(f"Error: Loaded checkpoint is not a dictionary (found type: {type(checkpoint)}). Cannot extract state_dict.")
        exit(1)

    if 'model' not in checkpoint:
        print("Error: Checkpoint dictionary does not contain the key 'model'.")
        print("Available keys:", checkpoint.keys())
        exit(1)
    # --- End Check ---

    state_dict = checkpoint['model']

    # Fix potential state_dict key mismatches (e.g., from DataParallel/DDP or torch.compile)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # Add other potential prefix fixes if needed (e.g., 'module.')
    unwanted_prefix_ddp = 'module.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix_ddp):
            state_dict[k[len(unwanted_prefix_ddp):]] = state_dict.pop(k)


    # Check if output layer size in checkpoint matches expected model_vocab_size
    output_head_key = 'lm_head.weight' # Adjust if your output layer name is different
    if output_head_key in state_dict:
        ckpt_vocab_size = state_dict[output_head_key].shape[0]
        if ckpt_vocab_size != model_vocab_size:
            print("\n!!! CRITICAL WARNING: Checkpoint Vocab Size Mismatch !!!")
            print(f"  Model configured with vocab_size = {model_vocab_size} (from meta.pkl)")
            print(f"  Checkpoint's output layer ('{output_head_key}') has size {ckpt_vocab_size}")
            print(f"  Loading WILL LIKELY FAIL or lead to unexpected behavior.")
            print(f"  Ensure meta.pkl reflects the *exact* vocab_size used for training this checkpoint.")
            # Consider exiting: exit(1)
    else:
        print(f"\nWarning: Could not find output layer key '{output_head_key}' in checkpoint to verify vocab size.")


    # Load the state dict using strict=False first for better debugging
    missing_keys, unexpected_keys = transformer_model.load_state_dict(state_dict, strict=False)

    # --- FIX: Initialize filtered_missing before the conditional block ---
    filtered_missing = []
    # --- END FIX ---

    if missing_keys:
        print("\nWarning: Missing keys when loading state_dict:")
        # Filter out expected buffer keys if desired
        expected_missing = {'attn.bias', 'attn.masked_bias'} # Add other known buffers if needed
        # Now this line just updates filtered_missing if there are missing keys
        filtered_missing = [k for k in missing_keys if not any(buf in k for buf in expected_missing)]
        if filtered_missing:
             for key in filtered_missing: print(f"  {key}")
        else:
             print("  (Only expected buffer keys like attn.bias seem missing)")

    if unexpected_keys:
        print("\nWarning: Unexpected keys when loading state_dict:")
        for key in unexpected_keys: print(f"  {key}")

    # This condition will now always work because filtered_missing is guaranteed to exist
    if not filtered_missing and not unexpected_keys:
        print("Model state_dict loaded successfully (ignoring expected buffers).")
    else:
         print("Model state_dict loaded with potential issues (see missing/unexpected keys above).")


except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {args.ckpath}")
    exit(1)
except pickle.UnpicklingError as e:
    print(f"Error: Failed to unpickle the checkpoint file at {args.ckpath}. It might be corrupted or saved incorrectly.")
    print(f"Unpickling error details: {e}")
    if "argparse.Namespace" in str(e):
        print("Hint: This often happens when loading older checkpoints with weights_only=True. Try weights_only=False (as implemented here), ensuring you trust the source.")
    exit(1)
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    print(f"Error type: {type(e).__name__}")
    exit(1)

transformer_model.eval() # Set model to evaluation mode (disables dropout, etc.)
transformer_model.to(device)

# --- Optional: Compile Model ---
if use_compile:
    print("\nCompiling model (takes a minute)...")
    try:
        transformer_model = torch.compile(transformer_model)
        print("Model compiled successfully.")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        use_compile = False # Fallback to non-compiled

# --- Generation Setup ---
# Encode the starting prompt
start_ids = encode(args.prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...] # Add batch dimension [1, T]

print(f"\nStarting generation with prompt: \"{args.prompt}\" ({len(start_ids)} tokens)")
print(f"Max new tokens: {args.max_tok}, Temperature: {args.temp}, Top-k: {args.top_k}")

# Set up context manager for mixed precision (if applicable and desired)
# BF16 is generally preferred on newer GPUs (Ampere+) if available
pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=pt_dtype)
print(f"Using autocast context with dtype: {pt_dtype if device=='cuda' else 'None (CPU)'}")
# ctx = nullcontext() # Uncomment this line to disable autocast if it causes issues

# --- Run Generation ---
print("Generating...")
with torch.no_grad(): # Crucial for inference efficiency
    with ctx: # Apply mixed precision context
        start_time = time.time()
        # Pass the actual tiktoken vocab size to the generate function
        # Ensure your generate function in model.py accepts these arguments
        y, kv_cache_size_gb = transformer_model.generate(
            idx=x,
            max_new_tokens=args.max_tok,
            temperature=args.temp,
            top_k=args.top_k,
            # Add the argument to handle vocabulary padding:
            tiktoken_vocab_size=tiktoken_vocab_size
        )
        elapsed = time.time() - start_time

# --- Decode and Print Output ---
generated_tokens = y[0].tolist() # Get tokens from the first batch element
generated_text = decode(generated_tokens) # Decode should work fine as padding tokens were avoided

print("\n--- Generated Text ---")
print(generated_text)
print("----------------------")

# Print generation statistics
num_generated = len(generated_tokens) - len(start_ids)
# Avoid division by zero if generation was instant or failed
tokens_per_sec = num_generated / elapsed if elapsed > 0 else float('inf')
print(f"\nGenerated {num_generated} tokens in {elapsed:.2f} seconds ({tokens_per_sec:.2f} tok/s)")
if kv_cache_size_gb is not None: # If generate returns KV cache size info
     print(f"Estimated final KV Cache size: {kv_cache_size_gb:.4f} GB")
