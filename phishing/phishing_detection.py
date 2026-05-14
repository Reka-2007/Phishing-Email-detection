
import pandas as pd
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

from scipy.sparse import hstack


df = pd.read_csv("email.csv")
df['label'] = df['label'].map({'safe': 0, 'phishing': 1})
print("Dataset Loaded Successfully!\n")
print(df.head())

def extract_url_features(text):
    urls = re.findall(r'(https?://\S+)', str(text))
    
    return {
        "url_count": len(urls),
        "has_ip_url": int(any(re.search(r'\d+\.\d+\.\d+\.\d+', url) for url in urls)),
        "has_suspicious_words": int(any(word in str(text).lower() 
                                        for word in ['login', 'verify', 'bank', 'urgent', 'password']))
    }

url_features = df['text'].apply(extract_url_features)
url_df = pd.DataFrame(url_features.tolist())
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_text = tfidf.fit_transform(df['text'])
X = hstack([X_text, url_df.values])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel Training Completed!\n")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()


def predict_email(email_text):
    url_feat = extract_url_features(email_text)
    url_feat_df = pd.DataFrame([url_feat])

    text_feat = tfidf.transform([email_text])
    combined = hstack([text_feat, url_feat_df.values])

    prediction = model.predict(combined)[0]
    return "Phishing" if prediction == 1 else "Safe"


sample_email = input("\nEnter an email message to test:\n")
result = predict_email(sample_email)
print("\nPrediction:", result)