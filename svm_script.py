from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
import pickle
import pandas as pd

Tfidf_vect = TfidfVectorizer(max_features=5000)
MODEL_PATH = 'SVM_MODEL/SVM_model.pickle'
DATASET_PATH = 'DATASET/RUcombined.csv'
def load_dataset():
    df = pd.read_csv(DATASET_PATH, encoding='ISO-8859-1', header = None)
    df = df.drop(0)
    df.rename(columns={0:'Reviews', 1:'Sentiment'}, inplace=True)
    df = df[df['Reviews'].notnull()]
    return df
df = load_dataset()
Tfidf_vect.fit(df['Reviews'])

with open(MODEL_PATH, 'rb') as handle:
  model = pickle.load(handle)


def chk_prob(doc_text):
  doc = Tfidf_vect.transform([doc_text])
  pol = model.predict_proba(doc)
  prob = {'Positive':pol.item(1), 'Negative':pol.item(0), 'Neutral':pol.item(2)}
  return prob

def chk_prob_max(doc_text):

  doc = Tfidf_vect.transform([doc_text])
  pol = model.predict_proba(doc)
  prob = {'Positive':pol.item(1), 'Negative':pol.item(0), 'Neutral':pol.item(2)}
  max_key = max(prob,key = prob.get)
  return max_key

print(chk_prob_max('Booooot maza aya hai'))
print(chk_prob_max('product kharab nikli hai'))
print(chk_prob_max('bohat khoob kya baat hai'))
print(chk_prob_max('maza aa gaya'))
print(chk_prob_max('yeah product bs shi hai'))
print(chk_prob_max('IS MOBILE KA GUZARA HAI'))

