# Spam Email Detection using Machine Learning
# Author: Mahfoud Slimen

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    "Win money now",
    "Limited offer just for you",
    "Meeting tomorrow at office",
    "Project deadline is next week",
    "Congratulations you won a prize",
    "Let's schedule a meeting"
]

labels = [1,1,0,0,1,0]  # 1 = Spam , 0 = Not Spam

# تحويل النصوص إلى أرقام
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

# تدريب النموذج
model = MultinomialNB()
model.fit(X_train, y_train)

# التنبؤ
predictions = model.predict(X_test)

# تقييم النموذج
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)
