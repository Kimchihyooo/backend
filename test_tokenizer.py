# test_tokenizer.py

# We import the exact same loader your app uses
from model_loader import get_model_loader
import sys

print("="*60)
print("--- 🚀 STARTING TOKENIZER TEST ---")
print("="*60)

try:
    # 1. Attempt to load all models
    print("[STEP 1/3] Calling get_model_loader() to load all models...")
    # This will print all the "Loading..." messages from your loader
    model_loader = get_model_loader()
    print("\n[SUCCESS] Model loading process finished.")
    
    # 2. Get the Filipino tokenizer from the loader
    print("[STEP 2/3] Accessing model_loader.tokenizer_tl...")
    tokenizer = model_loader.tokenizer_tl
    
    # 3. Check if it was loaded
    if tokenizer:
        print("[SUCCESS] Filipino RoBERTa tokenizer (tokenizer_tl) is LOADED!")
        
        # 4. Try to use it
        print("[STEP 3/3] Testing tokenizer on a sample sentence...")
        test_sentence = "Ito ay isang pagsubok."
        tokenized_output = tokenizer(test_sentence, return_tensors="pt")
        
        print("\n--- TOKENIZER OUTPUT (SUCCESS!) ---")
        print(tokenized_output)
        
        tokens = tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'][0])
        print("\n--- DECODED TOKENS ---")
        print(tokens)
        
    else:
        print("\n" + "!"*60)
        print("[FAILURE] model_loader.tokenizer_tl is 'None'.")
        print("This means the 'load_filipino_classification_models' function FAILED.")
        print("Please check the file path to your RoBERTa model in model_loader.py.")
        print("!"*60)

except Exception as e:
    print(f"\n[CRITICAL FAILURE] An error occurred during the test: {e}")
    print("This likely means there is an error in model_loader.py or a dependency.")
    sys.exit(1) # Exit with an error

print("\n" + "="*60)
print("--- ✅ TEST FINISHED ---")
print("="*60)