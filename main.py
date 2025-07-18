import pandas as pd
import numpy as np
from numpy.linalg import norm
import json, string
from collections import Counter

sentiment_Ratings = {
    "good": [4.0,5.0],
    "neutral":[3.0],
    "bad":[1.0,2.0]
}
stopwords = pd.read_csv("stopwords1.csv")
def main():
    documents = {} #{index: (rating,review as a list of terms)}
    i =0
    with open("5tester.json") as f:
        #read each json but only take the rating and the review
        for line in f:
            r = json.loads(line)
            rating = int(r.get('overall') or 0)
            text = preprocessing((r.get('reviewText', '')).strip())
            documents[i] = (rating,text)
            i+=1

        
    tfidf_vectors = calculateTFIDF(documents)
    print(calcualte_cosine_sim(documents, tfidf_vectors))

    
    
        
def preprocessing(text):
    #all lower case
    text = text.lower()

    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #remove stop words and words with numbers
    temp = text.split()
    temp = [w for w in temp if w not in stopwords.values and not any(c.isdigit() for c in w)]

    return temp

def calculateTFIDF(documents):
    N = len(documents)
    df = {}

    #Firstly getting how many documents the terms appear in  
    for i, (r, doc) in documents.items():
        terms = set(doc)
        for t in terms:
            if t not in df:
                df[t] = 1
            else:
                df[t] += 1


    #calculating the idf for each term
    idf ={}
    for t in df:
        idf[t] = np.log(N / df[t])

    all_tfidf_vactors = []
    allterms = np.sort(list(df.keys())) #alphabetize

    #creating tf-idf vectors for every document. Very sparse and filled with 0s
    for i, (r, doc) in documents.items():
        tf = Counter(doc) #calculates term frequency for doc
        doc_tfidf = []

        #I'm not in love with this method... The nested for-loop is taking a long time
        for term in allterms:
            tfidf_calc = tf[term] * idf[term] if term in tf else 0
            doc_tfidf.append(tfidf_calc)
        all_tfidf_vactors.append(doc_tfidf)

    return all_tfidf_vactors
            
#need to calcualte the simularity between two docs to get the weights 
def calcualte_cosine_sim(reviews,tfidf_v):
    #We build a NxN martix i think. That holds the sim score between every two documents
    N = len(tfidf_v)
    if N != len(reviews):
        print("SOMETHING WENT WRONG")
        return
    
    mat = []
    for i in range(N):
        row = []
        for j in range(N):
            if i==j: row.append(1)
            else:
                doc1 = tfidf_v[i]
                doc2 = tfidf_v[j]
                cos_sim = np.dot(doc1,doc2) /(norm(doc1)*norm(doc2))
                row.append(cos_sim)
        mat.append(row)
    
    return np.matrix(mat)


main()