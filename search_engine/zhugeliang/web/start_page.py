from flask import Flask, render_template
app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello, AI World!"


@app.route('/search/<query>')
def search(query):
    #return "Your search is: %s" % query
    return render_template('search.html', query=query)


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8000)
