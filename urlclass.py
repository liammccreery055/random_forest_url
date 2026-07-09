import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#load data set
#current_dir = os.path.dirname(os.path.abspath(__file__))

#csv_path = os.path.join(current_dir, 'phishing_site_urls.csv')

df = pd.read_csv('dsf.csv')

#data vectorization char_wb breaks urls into text chunks
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))
X = vectorizer.fit_transform(df['Column 1'].astype(str))
y = df['Column 2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=100, stratify=y
)
#train model
model = RandomForestClassifier(n_estimators=600, random_state=100, max_depth=20,min_samples_leaf=2,
                               min_samples_split=2,criterion="entropy", class_weight="balanced", max_features=0.3)
model.fit(X_train, y_train)

 
y_pred = model.predict(X_test)

#print training info
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, 'url_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
