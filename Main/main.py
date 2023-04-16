from flask import Flask, render_template, url_for, flash, redirect
from form import BookForm, UploadBook, DeleteBook
from recomm import recom
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
import os
import pandas as pd
import numpy as np
from flask_table import Table, Col

class Results(Table):
    id = Col('Id', show=False)
    title = Col('TOP RECOMMENDATIONS')

app=Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'
db=SQLAlchemy(app)


# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Construct the path to the file you want to access
dataset_path = os.path.join(parent_dir, 'dataset', 'Book.csv')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html',title="Home")


@app.route("/recommender",methods=['GET','POST'])
def recommender():
    form=BookForm()
    df=pd.read_csv(dataset_path)

    if form.bookname.data in list(df['Title']):
        flash(f'Here are the following recommendations for you', 'success')
        isbn=[]
        year=[]
        publisher=[]
        final_list=[]
        book=form.bookname.data
        output, index = recom(book)
        for i in index:
            isbn.append(df["ISBN"][i-1])
            year.append(df["Year"][i-1])
            publisher.append(df["Publisher"][i-1])
        for i in range(len(index)):
            temp=[]
            temp.append(output[i])
            temp.append(isbn[i])
            temp.append(year[i])
            temp.append(publisher[i])
            final_list.append(temp)

        return render_template('recommender.html',title='Recommender',form=form,final=final_list)
    else:
        flash(f'Name not clearly mentioned or does not exist in the database. Please try again.', 'danger')
        return redirect(url_for('recommender'))

    return render_template('recommender.html',title='Recommender',form=form)

@app.route("/uploadbook",methods=['GET','POST'])
def uploadbook():
    form=UploadBook()
    df=pd.read_csv('Book.csv')

    flash(f'Book Uploaded Succesfully', 'success')
    return redirect(url_for('home'))

    return render_template('uploadbook.html',title='Upload Book',form=form)


@app.route("/deletebook",methods=['GET','POST'])
def deletebook():
    form=DeleteBook()
    if form.validate_on_submit():
        flash(f'Book is Deleted', 'success')
        return redirect(url_for('home'))
    return render_template('deletebook.html',title='Delete Book',form=form)

if __name__ == '__main__':
    app.run(debug=True)