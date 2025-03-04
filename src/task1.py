import nltk
import re
import string
import numpy as np
import matplotlib.pyplot as plt
import gensim
import spacy
from nltk.corpus import stopwords, gutenberg
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine
from collections import Counter

# Load NLTK resources only once
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('gutenberg')
# nlp = spacy.load('en_core_web_sm')


def get_hamlet_corpus():
    hamlet_sents = gutenberg.sents('shakespeare-hamlet.txt')  # Tokenized sentences
    return [" ".join(sent) for sent in hamlet_sents]  # Join words into sentences


def get_top_words(corpus, top_n=100):
    all_words = []
    for sent in corpus:
        words = nltk.word_tokenize(sent.lower())  # Tokenize and lowercase
        words = [word for word in words if word.isalpha()]  # Remove punctuation
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    print([word for word, _ in word_freq.most_common(top_n)])
    return [word for word, _ in word_freq.most_common(top_n)]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


def prepare_dataset(corpus):
    return [preprocess_text(sentence) for sentence in corpus]


def train_word2vec(corpus, model_type="cbow", vector_size=100, window=5, min_count=1):
    sg = 1 if model_type == "skip-gram" else 0
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count, sg=sg)
    return model


def train_fasttext(corpus, vector_size=100, window=5, min_count=1):
    model = FastText(sentences=corpus, vector_size=vector_size, window=window, min_count=min_count)
    return model


def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)


def visualize_embeddings(model, words, method='pca', model_title=''):
    available_words = [word for word in words if word in model.wv]
    
    if not available_words:
        print("No words found in model's vocabulary.")
        return
    
    word_vectors = np.array([model.wv[word] for word in available_words])
    
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2)
        
    reduced_vectors = reducer.fit_transform(word_vectors)
    
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(available_words):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1])
        plt.annotate(label, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    model_title = model_title if model_title != '' else model.__class__.__name__
    plt.title(f'{model_title} visualization using {method.upper()}')
    plt.savefig(f'task1{model_title}.png')


if __name__ == "__main__":
    sample_corpus = get_hamlet_corpus()
    processed_data = prepare_dataset(sample_corpus)
    w2v_cbow = train_word2vec(processed_data, "cbow")
    w2v_sg = train_word2vec(processed_data, "skip-gram")
    fast_text = train_fasttext(processed_data)
    
    top_hamlet_words = get_top_words(sample_corpus)
    words_to_visualize = [word for word in top_hamlet_words if word in w2v_cbow.wv.index_to_key]
    words_to_visualize = words_to_visualize[:10]
    
    visualize_embeddings(w2v_cbow, words_to_visualize, model_title='cbow')
    visualize_embeddings(w2v_sg, words_to_visualize, model_title='sg')
    visualize_embeddings(fast_text, words_to_visualize)

