""" classify the given text based on the given corpus which contains  20 tags"""

"""Here TAG_WORDS are constants. That's why it is writtn in Uppercase"""

TAG_WORDS = ["research", "innovation", "education", "university", "students","faculty", "campus", "engineering", "medicine", "technology", "curriculum", "collaboration", "publication",
"laboratory", "scholarship", "mentorship", "internship", "entrepreneurship", "accreditation", "alumni"]

"""Here vec_list are constants. That's why it is written in Lowercase"""

#defined a fucntion called get_word_vectors which takes model, and tag words as input
def get_word_vectors(model, tags):

    # list to store words not found in the model
    dropped_words = []

    # list to store words found in the model
    word_vectors = []

    for tag in tags: 
        if tag in model.key_to_index:
            try:
                tv = model[tag]
                word_vectors.append(tv)
            except:
                #raise ValueError(f"Tag not in Vocabulary : {tag}")
                dropped_words.append(tag)
                word_vectors.append(None)

    return ((word_vectors, dropped_words))

def preprocess_text(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as file:
        raw_text = file.read()

    raw_tokens = raw_text.split(',')

    final_tokens = []
    STOPWORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "with", "at", "by", "from"}
    for token in raw_tokens:
        token = token.strip()

        if not token:
            continue

        lowercased_token = token.lower()
        cleaned_token = lowercased_token.strip('().;"\'')

        if cleaned_token and cleaned_token not in stop


 def test_empty_tags():
    #model = load_model
    result = get_word_vectors(model, [])

    #Test 1 : Test empty tags
    assert result == ([], [])

    #Test 2 : Test tags contain one word that does not exist in the model
    

        