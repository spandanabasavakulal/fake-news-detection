import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 Fake News Detection Dashboard")

# Load and merge datasets
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake['label'] = 0
    true['label'] = 1
    data = pd.concat([fake[['text', 'label']], true[['text', 'label']]], axis=0)
    return data.sample(frac=1).reset_index(drop=True)

data = load_data()

# Vectorizer and model
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X = tfidf.fit_transform(data['text'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Sidebar: Pie Chart + Accuracy
st.sidebar.subheader("📊 Label Distribution")
label_counts = data['label'].value_counts()
labels = ['Real', 'Fake']
sizes = [label_counts[1], label_counts[0]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
ax1.axis('equal')
st.sidebar.pyplot(fig1)
st.sidebar.success(f"✅ Accuracy: {round(acc * 100, 2)}%")

# Main layout: Charts
st.subheader("📊 Count of Real vs Fake News (Bar Chart)")
fig2 = plt.figure()
sns.countplot(data=data, x='label', palette=['red', 'green'])
plt.xticks([0, 1], ['Fake', 'Real'])
st.pyplot(fig2)

st.subheader("📈 Length of News Text (Histogram)")
data['text_length'] = data['text'].apply(len)
fig3 = plt.figure()
sns.histplot(data, x='text_length', hue='label', bins=50, palette=['red', 'green'])
plt.title("Distribution of News Text Length")
st.pyplot(fig3)

st.subheader("☁️ Word Cloud of News Content")
real_text = " ".join(data[data["label"] == 1]["text"].tolist())
fake_text = " ".join(data[data["label"] == 0]["text"].tolist())

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Real News Word Cloud**")
    real_wc = WordCloud(width=400, height=300, background_color='white').generate(real_text)
    st.image(real_wc.to_array())

with col2:
    st.markdown("**Fake News Word Cloud**")
    fake_wc = WordCloud(width=400, height=300, background_color='white').generate(fake_text)
    st.image(fake_wc.to_array())

# Prediction Section
st.subheader("🖊️ Enter a News Headline or Article to Predict")

user_input = st.text_area("Type or paste news content here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = tfidf.transform([user_input])
        prediction = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        if prediction == 1:
            st.success("✅ It's Real News")
            st.info(f"Confidence: {round(proba[1]*100, 2)}%")
        else:
            st.error("❌ It's Fake News")
            st.info(f"Confidence: {round(proba[0]*100, 2)}%")
