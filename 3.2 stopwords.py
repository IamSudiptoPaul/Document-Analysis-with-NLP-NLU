import nltk
from nltk.corpus import stopwords

# may need to download the list first
nltk.download('stopwords')

# Load the standard English stop words
stop_words_list = stopwords.words('english')

print(f"Total number of English stop words: {len(stop_words_list)}")
print("\nFirst 20 stop words being filtered from the social media posts:")
print(stop_words_list[:20])

# To check if a specific word from dataset is a stop word:
word_to_check = "the"
if word_to_check in stop_words_list:
    print(f"\nResult: '{word_to_check}' is a stop word and was removed during preprocessing.")