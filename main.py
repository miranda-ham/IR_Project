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
        #read each json but only take the rating and the review
        for line in f:
            r = json.loads(line)
            #I'm assuming "overall" and "reviewText" will always be there - should put checks in later?
            #or user can input what the columns names are, so diff files can be inputted
            rating = int(r.get('overall') or 0)
            text = preprocessing((r.get('reviewText', '')).strip())
            
            #places all documents as strings with their respective rating 
            #not sure if best way to store info yet
            if rating not in reviews:
                reviews[rating] = [text]
            else:
                reviews[rating].append(text)
        
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

    #Firstly getting how many documents the terms appear in
    for doc in documents:
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
    allterms = np.sort(list(df.keys()))

    #creating tf-idf vectors for every document. Very sparse and filled with 0s
    for doc in documents:
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
    pass


main()