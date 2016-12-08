import os
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

yesorno = "YES"
fileurl = "./latest/small-set/train/" + yesorno + "/"
listOfFiles = os.listdir(fileurl)
textlist = []
stopset = set(stopwords.words('english'))

print("Running")

while len(listOfFiles) > 0:
    filename = listOfFiles.pop(0)
    if (filename != ".DS_Store"):
        with open(fileurl + filename) as f:
            file = f.read() # returns a string of document
            file = file.replace("\xc2\xa0", " ").lower().decode("utf-8", "replace")
            textlist.append(file)

vectorizer = TfidfVectorizer(max_df = 0.5, stop_words = stopset, use_idf = True, ngram_range = (1, 3))
X = vectorizer.fit_transform(textlist)
# X is a sparse Tf-idf matrix
print("done")

lsa = TruncatedSVD(n_components = 20, n_iter = 20)
print("done2")
lsa.fit(X)
print("done3")

terms = vectorizer.get_feature_names()
print(terms)
text_file = open(yesorno + "words.txt", "w")
for i, comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    print(len(termsInComp))
    sortedTerms = sorted(termsInComp, key = lambda x : x[1], reverse = True) [:5]
    print ("Concept %d:" % i)
    text_file.write("Concept %d:" % i + "\n")
    for term in sortedTerms:
        print (term[0])
        text_file.write(term[0] + "\n")
    print(" ")
text_file.close()
