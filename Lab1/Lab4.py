import nltk 
nltk.download('stopwords')
nltk.download('punkt')

text = """decision trees are a very effective method of supervised learning. it aims is the partition of a dataset into groups as homogeneous as possible in terms of the variable to be predicted. it takes as input a set of classified data, and outpus a tree that resembles to an orientation diagram where each end node leaf is a decision a class and each non final node internal represents a test. each leaf represents the decision of belonging to a class of data verifying all tests path from the root to the leaf"""


from nltk.corpus import stopwords 
stop_words = set(stopwords.words("english"))


from nltk.tokenize import word_tokenize,sent_tokenize
tokenize_words = word_tokenize(text)

tokenize_words_without_stop_words = []
for word in tokenize_words:
    if word not in stop_words:
        tokenize_words_without_stop_words.append(word)

print("Cac tu dung duoc xoa:\n",set(tokenize_words)-set(tokenize_words_without_stop_words))

print("Khong phai tu dung:\n",tokenize_words_without_stop_words)