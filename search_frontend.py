import math
import pickle
from collections import Counter
import os
import pickle

from flask import Flask, request, jsonify
import re
import nltk
from nltk.corpus import stopwords
from inverted_index_gcp2 import *




class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)


def tokenize(text):
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    filter_tokens = []
    for token in tokens:
        if token in all_stopwords:
            continue
        else:
            filter_tokens.append(token)
    return filter_tokens



inverted = InvertedIndex()
dl = inverted.read_index('dl','dl')
inverted = InvertedIndex()
dt = inverted.read_index('dt','dt')

all_tokens = 0
for tokens_len in dl.values():
    all_tokens += tokens_len
all_doc = len(dl.keys())




inverted = InvertedIndex()
inverted_text = inverted.read_index('postings_gcp_text', 'index')

inverted = InvertedIndex()
inverted_title = inverted.read_index('postings_gcp_title', 'index')

inverted = InvertedIndex()
inverted_anchor = inverted.read_index('postings_gcp_anchor', 'index')



def BM25_text_search(query):
    # BEGIN SOLUTION
    query = tokenize(query)
    score = Counter()
    dl_len = len(dl)
    b = 0.75
    k1 = 2
    avgdl = all_tokens / all_doc
    for term in query:
        posting_list = inverted_text.read_posting_list(term, 'postings_gcp_text')
        if posting_list:
            term_df = inverted_text.df[term]
            idf = math.log((dl_len - term_df + 0.5) / (term_df + 0.5) + 1)
            for id, freq in posting_list:
                if id in dl.keys():
                    pl = dl[id]
                    numerator = freq * (k1 + 1)
                    denominator = (freq + k1 * (1 - b + (b * (pl / avgdl))))
                    score[id] += idf * (numerator / denominator)

    most_score_text = score.most_common(100)
    res_text=[]
    for id, value in most_score_text:
        res_text.append((id, dt[id]))

    # END SOLUTION
    return res_text




def title_rank(query):
    title_res = []
    query = tokenize(query)
    postings_lists = []
    for term in query:
        posting_lists = inverted_title.read_posting_list(term, 'postings_gcp_title')
        postings_lists.append(posting_lists)

    id_tf = [(id,tf) for pl in postings_lists for id, tf in pl]
    sorted_id_tf = Counter(id_tf).most_common()
    res_title_tf={}
    for id_,tf_ in id_tf:
       if id_ in dt.keys():
          res_title_tf[id_]=tf_/len(dt[id_])  
    return Counter(res_title_tf).most_common(100)



def anchor_rank(query):

    anchor_res = []
    query = tokenize(query)
    postings_lists = []
    for term in query:
        posting_lists = inverted_anchor.read_posting_list(term, 'postings_gcp_anchor')
        postings_lists.append(posting_lists)

    id_tf = [(id,tf) for pl in postings_lists for id, tf in pl]
    sorted_id_tf = Counter(id_tf).most_common()
    res_anchor_tf={}
    for id_,tf_ in id_tf:
       if id_ in dt.keys():
          res_anchor_tf[id_]=tf_/len(dt[id_])  
    return Counter(res_anchor_tf).most_common(100)


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res1=BM25_text_search(query)
    res2=anchor_rank(query)
    res3 = title_rank(query)
    id_ranking = Counter()

    for i in range(len(res1)):
        id_ranking[res1[i][0]] += 100 / (i + 1)

    for j in range(len(res2)):
        id_ranking[res2[j][0]] += 100*res2[j][1]

    for j in range(len(res3)):
        id_ranking[res3[j][0]] += 100*res3[j][1]


    for id, value in id_ranking.most_common(100):
        res.append((str(id), dt[id]))

    # END SOLUTION
    return jsonify(res)




@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
