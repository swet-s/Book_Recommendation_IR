import nltk
import numpy as np
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def recom(books_user_likes):
    def get_title_from_index(index):
        return book_data[book_data.index == index]["Title"].values[0]

    def get_index_from_title(Title):
        return book_data[book_data.Title == Title]["index"].values[0]

    book_data = pd.read_csv(r"C:\Projects\python\book_recom\dataset\Book.csv")
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

    # print('Recommended books for "{}":'.format(books_user_likes))
    # for book in top_books:
    #     print(book)

    # #Get index of this book from its title
    #	books_index = get_index_from_title(books_user_likes)
    #	similar_books = list(enumerate(similarity_matrix[books_index]))

    # #Get a list of similar books in descending order of similarity score
    #	sorted_similar_books = sorted(similar_books,key=lambda x:x[1],reverse=True)
    #
    # # titles of first 50 books
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
    books = pd.read_csv("C:\\Projects\\python\\book_recom\\dataset\\Bookz.csv")
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
