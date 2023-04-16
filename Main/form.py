from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError


class BookForm(FlaskForm):
    bookname = StringField('Enter Book Name', validators=[DataRequired()])
    submit = SubmitField('Get Recommendations !')


class UploadBook(FlaskForm):
    ISBN = StringField('ISBN', validators=[DataRequired(), Length(min=2, max=15)])
    Title = StringField('Title', validators=[DataRequired()])
    Author = StringField('Author', validators=[DataRequired()])
    Publisher = StringField('Publisher', validators=[DataRequired()])
    ImageURL = StringField('Image URL', validators=[DataRequired()])
    submit = SubmitField('Upload Details')


class DeleteBook(FlaskForm):
    ISBN = StringField('Enter ISBN :', validators=[DataRequired(), Length(min=2, max=15)])
    submit = SubmitField('Delete Book ;)')
