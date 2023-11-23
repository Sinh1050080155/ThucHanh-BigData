
from pybloom_live import BloomFilter
import nltk

nltk.download('words')

file_path = "wordcount.txt"

with open(file_path, 'r') as file:
    text = file.read()

words = nltk.word_tokenize(text)

bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)

for word in words:
    bloom_filter.add(word)

words_to_check = ["of", "example", "some", "words", "to", "check"]


for check in words_to_check:
    if check in bloom_filter:
        print(f"Từ '{check}' Có trong văn bản.")
    else:
        print(f"Từ '{check}' Không có trong văn bản.")