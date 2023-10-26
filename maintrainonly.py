from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Open training data and split into sentences
data = []
with open("train.txt", 'r') as file:
    sentence = []
    for line in file:
        if line.strip():
            word, pos_tag, chunk_tag = line.strip().split()
            sentence.append([word, pos_tag])
        else:
            data.append(sentence)
            sentence = []

def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = ["a", "e", "i", "o", "u"]
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    return count if count != 0 else 1

def feature_extraction(sentence, i):
    word = sentence[i][0]
    last_word = sentence[i-1][0] if i > 0 else '<START>'
    next_word = sentence[i+1][0] if i < len(sentence) - 1 else '<END>'
    distance_from_end = len(sentence) - i - 1
    return {
        # 'word' : word,
        'word_length' : len(word), # Get length of each word
        'is_capitalized' : word[0].isupper(), # Get capitalization of each word
        'has_special' : 1 if (not char.isalpha() for char in word) else 0, # Get if any digits in word
        'prefix2' : word[:3], # Get next possible prefix
        'prefix3' : word[:4], # Get last possible prefix
        'suffix1' : word[-1:], # Get possible suffix
        'suffix2' : word[-2:], # Get next possible suffix
        'suffix3' : word[-3:], # Get last possible suffix
        'last_word' : last_word, # Get word that came before
        'next_word' : next_word, # Get word that comes next
    }

# Extract features for each word within all sentences
X, y = [], []

for sentence in data:
    for i in range(len(sentence)):
        X.append(feature_extraction(sentence, i))
        y.append(sentence[i][1])

# SKLearn DictVectorizer to switch features to matrix
dict_vectorizer = DictVectorizer(sparse=True)
X_transformed = dict_vectorizer.fit_transform(X)

lr_classifier = LogisticRegression(max_iter=1000, random_state=42, solver='sag', n_jobs=-1, verbose=1)
lr_classifier.fit(X_transformed, y)

unlabeled = []
with open("in_domain_test_without_label.txt") as file:
    sentence = []
    for line in file:
        if line.strip():
            word = line.strip()
            sentence.append(word)
        else:
            unlabeled.append(sentence)
            sentence = []

X_unlabeled = []
for sentence in unlabeled:
    for i in range(len(sentence)):
        X_unlabeled.append(feature_extraction(sentence, i))

X_unlabeled_transformed = dict_vectorizer.transform(X_unlabeled)

y_pred_unlabeled = lr_classifier.predict(X_unlabeled_transformed)

with open("hundred.txt", 'w') as file:
    idx = 0
    for sentence in unlabeled:
        for word in sentence:
            file.write(f"{word}\t{y_pred_unlabeled[idx]}\n")
            idx += 1
        file.write("\n")
