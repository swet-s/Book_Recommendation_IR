import nltk
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# def boolean_retrieval(books_user_likes, book_data):
#
#     # Preprocess the book data
#     stop_words = set(stopwords.words('english'))
#     stemmer = PorterStemmer()
#
#     book_data['Title'] = book_data['Title'].apply(lambda x: x.lower())
#     book_data['Title'] = book_data['Title'].apply(
#         lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x) if word not in stop_words]))
#     book_data['Author'] = book_data['Author'].apply(lambda x: x.lower())
#     book_data['Author'] = book_data['Author'].apply(
#         lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x) if word not in stop_words]))
#
#     index = {}
#
#     for i, row in book_data.iterrows():
#         for feature in ['Title', 'Author']:
#             for word in word_tokenize(row[feature]):
#                 if word not in index:
#                     index[word] = {i}
#                 else:
#                     index[word].add(i)
#
#     # Define a function to search the index using boolean operators
#     def boolean_search(books_user_likes):
#         query = books_user_likes.lower()
#         terms = word_tokenize(books_user_likes)
#         doc_set = set(range(len(book_data)))
#
#         # Process each term in the query
#         for term in terms:
#             if term == 'and':
#                 continue
#             elif term == 'or':
#                 continue
#             elif term == 'not':
#                 continue
#             else:
#                 if term in index:
#                     doc_set = doc_set.intersection(index[term])
#                 else:
#                     doc_set = set()
#                     break
#
#         # Process the boolean operators in the query
#         for i in range(1, len(terms), 2):
#             operator = terms[i]
#             term = terms[i + 1]
#             if term in index:
#                 if operator == 'and':
#                     doc_set = doc_set.intersection(index[term])
#                 elif operator == 'or':
#                     doc_set = doc_set.union(index[term])
#                 elif operator == 'not':
#                     doc_set = doc_set.difference(index[term])
#
#         return doc_set
#
#     def recommend_books(books_user_likes, num_recommendations=10):
#         matching_docs = boolean_search(books_user_likes)
#         relevance_scores = []
#         for i in matching_docs:
#             doc = book_data.iloc[i]
#             relevance_scores.append((i, len(set(books_user_likes.split()) & set(doc['Author'].split()))))
#         relevance_scores = sorted(relevance_scores, key=lambda x: x[1], reverse=True)
#         top_books = []
#         for i in range(min(num_recommendations, len(relevance_scores))):
#             top_books.append(book_data.iloc[relevance_scores[i][0]]['Title'])
#
#         return top_books
#
#     # Define a function to recommend books
#     top_books = recommend_books(books_user_likes, 10)
#
#     return top_books

def vector_space(books_user_likes, book_data):
    books = []

    # df=books
    # img=pd.read_csv(r"C:\Projects\python\book_recom\dataset\Imagez.csv")

    wordnet_lemmatizer = WordNetLemmatizer()

    # features = ['Title','Author','Publisher']
    # for feature in features:
    #     book_data[feature] = book_data[feature].fillna('')
    #
    #
    # def combine_features(row):
    #     try:
    #         return row['Title'] +" "+row['Author']+" "+row['Publisher']
    #     except:
    #         print("Error:", row)

    # book_data["combined_features"] = book_data.apply(combine_features,axis=1)

    for text in book_data['Title']:
        tokens = word_tokenize(text.lower())

        lemmatized = [wordnet_lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]

        clean_text = ' '.join([token for token in lemmatized])

        books.append(clean_text)

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(books)

    # print(feature_matrix)

    # Calculate cosine similarity between each pair of books
    similarity_matrix = cosine_similarity(feature_matrix)

    # print(similarity_matrix)

    query_vec = vectorizer.transform([books_user_likes])
    scores = cosine_similarity(query_vec, feature_matrix)
    top_books = [book_data['Title'][i] for i in scores.argsort()[0][::-1]]

    return top_books


def recom(books_user_likes):
    def get_title_from_index(index):
        return book_data[book_data.index == index]["Title"].values[0]

    def get_index_from_title(Title):
        return book_data[book_data.Title == Title]["index"].values[0]

    book_data = pd.read_csv(r"C:\Projects\python\book_recom\dataset\Book.csv")

    # top_books = boolean_retrieval(books_user_likes, book_data)
    top_books = vector_space(books_user_likes, book_data)

    l = []
    t = []
    i = 0
    for element in top_books:
        print(element)

        l.append(element)
        t.append(int(get_index_from_title(l[i])))
        i = i + 1
        if i > 9:
            break

    output = l
    index = t

    print(index)

    # imgg = []
    year = []
    author = []
    final_list = []
    for i in index:
        # imgg.append(img["Image-URL-M"][i-1])
        year.append(book_data["Year"][i - 1])
        author.append(book_data["Author"][i - 1])
    for i in range(len(index)):
        temp = []
        temp.append(output[i])
        # temp.append(imgg[i])
        temp.append(year[i])
        temp.append(year[i])
        temp.append(author[i])
        final_list.append(temp)

    return final_list


def bookdisp():
    books = pd.read_csv("C:\\Projects\\python\\book_recom\\dataset\\Book.csv")
    img = pd.read_csv("C:\\Projects\\python\\book_recom\\dataset\\Imagez.csv")

    title = []
    imgg = []
    year = []
    author = []
    finallist = []

    r = np.random.randint(2, 1000, 10)

    for i in r:
        title.append(books["Title"][i - 1])
        imgg.append(img["Image-URL-M"][i - 1])
        year.append(books["Year"][i - 1])
        author.append(books["Author"][i - 1])

    for i in range(10):
        temp = []
        temp.append(title[i])
        temp.append(imgg[i])
        temp.append(year[i])
        temp.append(author[i])
        finallist.append(temp)

    return finallist
