import string

def calculate_vocabulary_count(file_path):
    try:
        # Open and read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Split the text by commas since your data is comma-separated
        raw_tokens = content.split(',')
        
        cleaned_words = []
        for token in raw_tokens:
            # Strip whitespace and common NLP brackets/punctuation from edges
            word = token.strip().strip('().,;"\'').lower()
            
            # Ensure the token isn't empty after cleaning
            if word:
                cleaned_words.append(word)
        
        # Using a set to keep only unique words
        vocabulary = set(cleaned_words)
        
        # Output results
        print(f"Total Words (Tokens) found: {len(cleaned_words)}")
        print(f"Unique Vocabulary Count: {len(vocabulary)}")
        return len(vocabulary)

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the function for 'words.txt'
calculate_vocabulary_count(r'C:\Users\bda\Desktop\261131\ALA_Lab\Sep_21\words.txt')
