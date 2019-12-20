import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix , classification_report  ,accuracy_score

data = pd.read_csv("Restaurant_Reviews.csv" , sep='\t')
print(data)

wordnet = WordNetLemmatizer()

corpus = []
for i in range(0 , len(data)):
    review = re.sub('[^a-zA-Z]' , ' ' , data["Review"][i])
    review = review.lower()
    review = review.split()

    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)

    corpus.append(review)

print(corpus)


cv = TfidfVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

print(X)
y = data.iloc[:,1].values

print(y)

train_x , test_x ,train_y , test_y = train_test_split(X,y,test_size=1/3 , random_state=0)

model = MultinomialNB()
model.fit(train_x , train_y)

pred_y = model.predict(test_x)

print(confusion_matrix(test_y , pred_y))
print(classification_report(test_y , pred_y))
print(accuracy_score(test_y , pred_y))

