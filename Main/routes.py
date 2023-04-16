from flask import render_template, url_for, flash, redirect
from Main.form import BookForm, UploadBook, DeleteBook
from Main.recomm import recom, bookdisp
from Main import app
import pandas as pd
import csv
from csv import writer
import os

# Get the absolute path of the directory containing the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# Construct the path to the file you want to access
dataset_path = os.path.join(parent_dir, 'dataset', 'Book.csv')
image_path = os.path.join(parent_dir, 'dataset', 'Imagez.csv')

@app.route("/")
@app.route("/home")
def home():
    list1 = bookdisp()
    return render_template('home.html', content=list1)


@app.route("/recommender", methods=['GET', 'POST'])
def recommender():
    form = BookForm()
    # df = pd.read_csv("C:\\Projects\\python\\book_recom\\dataset\\Book.csv")
    if form.validate_on_submit():
        flash(f'Here are the following recommendations for you', 'success')
        book = form.bookname.data
        final_list = recom(book)
        return render_template('recommender.html', title='Recommender', form=form, final=final_list)
    return render_template('recommender.html', title='Recommender', form=form)


def upload(file_name, list_of_elem):
    with open(file_name, 'a+') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


@app.route("/uploadbook", methods=['GET', 'POST'])
def uploadbook():
    form = UploadBook()
    df = pd.read_csv(dataset_path)
    if form.validate_on_submit():
        i = max(df['index'] + 1)
        li = [i, i, form.ISBN.data, form.Title.data, form.Author.data, form.Publisher.data]
        upload(dataset_path, li)
        flash(f'Book Uploaded Succesfully', 'success')
        return redirect(url_for('home'))
    return render_template('uploadbook.html', title='Upload Book', form=form)


def delete(isbn_num, file_name):
    lines = list()
    with open(file_name, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            lines.append(row)
            for field in row:
                if field == isbn_num:
                    lines.remove(row)
    with open(file_name, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


@app.route("/deletebook", methods=['GET', 'POST'])
def deletebook():
    form = DeleteBook()
    if form.validate_on_submit():
        delete(form.ISBN.data, dataset_path)
        flash(f'Book is Deleted', 'success')
        return redirect(url_for('home'))
    return render_template('deletebook.html', title='Delete Book', form=form)
