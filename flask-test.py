from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>I am a complete beginner in Flask!</p>"


@app.route("/about/<username>")
def about_page(username):
    return f"<h2>This page is about {username}</h2>"