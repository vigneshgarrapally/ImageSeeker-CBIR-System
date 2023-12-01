from flask import Blueprint, render_template

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/results')
def results():
    # Logic for CBIR goes here
    return render_template('results.html')
