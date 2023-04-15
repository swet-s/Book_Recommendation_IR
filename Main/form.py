from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from flask_login import current_user
from wtforms import StringField, PasswordField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length,Email,EqualTo, ValidationError


class BookForm(FlaskForm):
	bookname=StringField('Enter Book Name',validators=[DataRequired()])
	submit=SubmitField('Get Recommendations !')

class UploadBook(FlaskForm):
	ISBN=StringField('ISBN',validators=[DataRequired(),Length(min=2,max=15)])
	Title=StringField('Title',validators=[DataRequired()])
	Author=StringField('Author',validators=[DataRequired()])
	Publisher=StringField('Publisher',validators=[DataRequired()])
	ImageURL=StringField('Image URL',validators=[DataRequired()])
	submit=SubmitField('Upload Details')


class Contact(FlaskForm):
	subject=StringField('Subject',validators=[DataRequired(),Length(min=5,max=12)])
	query=StringField('Query',validators=[DataRequired()])
	submit=SubmitField('Submit')


class DeleteBook(FlaskForm):
	ISBN=StringField('Enter ISBN :',validators=[DataRequired(),Length(min=2,max=15)])
	submit=SubmitField('Delete Book ;)')

