from flask import Flask,request,render_template

from indexing_and_searching.search import Searcher
app = Flask(__name__)

searcher = Searcher()

@app.route('/search',methods = ['get'])
def index():

    return render_template('search.html',use_clf=True)

@app.route('/search',methods = ['post'])
def search():
    command = request.form['query']
    use_clf = ("use_clf" in request.form)

    if(use_clf):
        results, probs, key_use= searcher.search(command,100,use_clf)
        return render_template('search.html', results=results, use_clf=use_clf, command=command,probs=probs, key_use=key_use)
    else:
        results = searcher.search(command,100,use_clf)
        return render_template('search.html', results=results,use_clf=use_clf,command=command)


if __name__ == '__main__':
    app.debug = True
    app.run()