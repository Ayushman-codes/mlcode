import itertools
import sys

print("Checking dictionary corpus...")
import nltk

try:
    from nltk.corpus import words
except LookupError:
    print("Downloading NLTK words database (this may take a moment)...")
    nltk.download('words', quiet=True)
    from nltk.corpus import words

def find_valid_words():
    sys.stdout.flush()
    user_input = input("\nEnter 5 letters (e.g., a e r t p or aertp): ")
    letters = [char.lower() for char in user_input if char.strip()]
    
    if len(letters) != 5:
        print(f"Error: You entered {len(letters)} letters. Please provide exactly 5 letters.")
        return

    print("Checking combinations against dictionary...")
    english_vocab = set(words.words())
    unique_perms = set(itertools.permutations(letters))
    
    valid_words = set()
    for perm in unique_perms:
        word = "".join(perm)
        if word in english_vocab:
            valid_words.add(word)
            
    print(f"\nFound {len(valid_words)} valid meaningful word(s):\n")
    if valid_words:
        for word in sorted(valid_words):
            print(word)
    else:
        print("No valid English words found for this combination.")

if __name__ == "__main__":
    find_valid_words()