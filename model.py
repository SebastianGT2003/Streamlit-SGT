import re
import string
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem import SnowballStemmer

class Model:
    def __init__(self, data, column_name):
        self.column_name = column_name
        self.data = data

        self.spanish_stopwords = set(stopwords.words('spanish'))
        self.spanish_stemmer = SnowballStemmer("spanish")

    def processing(self):
        self.data.head()
        self.data["clean_text"] = self.data[self.column_name].apply(self.cleaning)
        self.data["clean_text"] = self.data["clean_text"].apply(self.clean_with_stopwords_and_stemming_regex)

    def clean_with_stopwords_and_stemming_regex(self, text):
        stopwords_es = set(stopwords.words('spanish'))
        stemmer_es = SnowballStemmer("spanish")
        tokens = re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)
        stemmed_tokens = [stemmer_es.stem(token) for token in tokens if token not in stopwords_es]
        return " ".join(stemmed_tokens).strip()

    def cleaning(self, text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\w*\d\w*', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.strip()
        return text

    def wordcloud(self):
        plt.style.use('default')
        text = " ".join(review for review in self.data["clean_text"])
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400).generate(text)
        figure, ax = plt.subplots(facecolor='white')
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title("Word Cloud - Clean Reviews", color="#222", fontsize=16)
        figure.patch.set_facecolor('white')
        # Marco negro
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#222')
            spine.set_linewidth(2)
        return figure

    def bar_chart(self):
        plt.style.use('default')
        full_text = ' '.join(self.data["clean_text"].str.lower())
        tokens = re.findall(r'\b\w+\b', full_text)
        count = Counter(tokens)
        words, freqs = zip(*count.most_common(10))
        figure, ax = plt.subplots(facecolor='white')
        ax.barh(words[::-1], freqs[::-1], color="#222")
        ax.set_title("Top 10 Most Frequent Words", color="#222", fontsize=14)
        ax.set_xlabel("Frequency", color="#222")
        ax.set_ylabel("")
        ax.tick_params(colors="#222")
        figure.tight_layout()
        figure.patch.set_facecolor('white')
        # Marco negro
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#222')
            spine.set_linewidth(2)
        return figure

    def classification(self, emotion_model):
        self.data["classification"] = self.data["clean_text"].apply(lambda x: emotion_model(x)[0]['label'])
        self.data["classification"] = self.data["classification"].astype("category")

    def question(self, tokenizer, model, question_text):
        context = self.data[self.column_name]
        prompt = f"question: {question_text}  context: {context}"
        inputs = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt").to(model.device)
        ids = model.generate(**inputs, max_length=64, num_beams=4)
        answer = tokenizer.decode(ids[0], skip_special_tokens=True)
        return answer

    def results_chart(self):
        plt.style.use('default')
        counts = self.data["classification"].value_counts()
        figure, ax = plt.subplots(facecolor='white')
        counts.plot(kind='bar', ax=ax, color="#222")
        ax.set_title("Review Classification", color="#222", fontsize=14)
        ax.set_xlabel("Classification", color="#222")
        ax.set_ylabel("Frequency", color="#222")
        ax.tick_params(colors="#222")
        figure.patch.set_facecolor('white')
        # Marco negro
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#222')
            spine.set_linewidth(2)
        return figure