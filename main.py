import pandas as pd
import numpy as np
import json, string
from collections import Counter

sentiment_Ratings = {
    "good": [4.0,5.0],
    "neutral":[3.0],
    "bad":[1.0,2.0]
}
documents = []
stopwords = pd.read_csv("stopwords1.csv")

def main():
    reviews = {}
    with open("AMAZON_FASHION_5.json") as f:
        for line in f:
            r = json.loads(line)
            rating = int(r.get('overall') or 0)
            text = preprocessing((r.get('reviewText', '')).strip())
            if rating not in reviews:
                reviews[rating] = [{text: 0}]
            else:
                reviews[rating].append({text:0})
        
    tfidf_vectors = calculateTFIDF()
    calcualte_cosine_sim(reviews, tfidf_vectors)
    print("wait")
    
    
        
def preprocessing(text):
    #all lower case
    text = text.lower()

    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    #remove stop words and words with numbers
    temp = text.split()
    temp = [w for w in temp if w not in stopwords.values and not any(c.isdigit() for c in w)]
    documents.append(temp)
    text = " ".join(temp)

    return text

def calculateTFIDF():
    N = len(documents)
    df = {}
    for doc in documents:
        terms = set(doc)
        for t in terms:
            if t not in df:
                df[t] = 1
            else:
                df[t] += 1
    
    idf ={}
    for t in df:
        idf[t] = np.log(N / df[t])

    all_tfidf_vactors = []
    allterms = np.sort(list(df.keys()))
    for doc in documents:
        #we have a doc with terms
        #now we need to get tf for every word which is 
        tf = Counter(doc)
        doc_tfidf = []
        for term in allterms:
            tfidf_calc = tf[term] * idf[term] if term in tf else 0
            doc_tfidf.append(tfidf_calc)
        all_tfidf_vactors.append(doc_tfidf)

    return all_tfidf_vactors
            
def calcualte_cosine_sim(reviews,tfidf_v):
    pass


main()