from flask import Flask, render_template, request
from postvision import *

app = Flask(__name__)

pvi = PostVisionInstance()

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/direct')
def direct_query():
    '''
    Not Dependent on the Web interface, Returns a Json String with the information
    '''
    response = {}
    teamname = request.args.get('teamname')
    response['teamname'] = teamname

    k = 5
    query_result = pvi.execute_query(teamname, k)

    for k, v in query_result.items():
        response[k] = v

    return response

if __name__ == '__main__':
    app.run(debug=True)