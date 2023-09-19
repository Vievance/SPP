from app import app, db
from flask import render_template, flash, redirect, url_for, request
from app.forms import LoginForm, RegistrationForm
from flask_login import current_user, login_user, logout_user, login_required
from app.models import User
from werkzeug.urls import url_parse

books = [
    {
        'title': 'Memories, Dreams, Reflections',
        'author': 'Carl Jung',
        'ISBN': '9780006540274',
        'genre': 'non-fiction'
    },
    {
        'title': 'KHARIS: Hellenic Polytheism Explored',
        'author': 'Sarah Kate Istra Winter',
        'ISBN': '9781089224457',
        'genre': 'non-fiction'
    }
]

@app.route('/')
@app.route('/home')
@app.route('/index')
def home():
    return render_template('home.html', title='Home', books=books)

@app.route('/about')
def about():
    return render_template('about.html', title='About', books=books)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('home')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/secret')
@login_required
def secret():
    return render_template("secret.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)