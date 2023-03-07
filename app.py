from flask import Flask, render_template, request, url_for, redirect
from postvision import *

app = Flask(__name__)

pvi = PostVisionInstance()

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        teamname = request.form['teamname']
        return redirect(url_for('result', name=teamname))
    return render_template('home.html')

@app.route('/result/<name>', methods=['POST', 'GET'])
def result(name):
    
    k = 5
    query_result = pvi.execute_query(name, k)
    query_result['teamname'] = name

    return render_template('result.html', query_result=query_result)

@app.route('/direct/<name>')
def direct_query(name):
    '''
    Not Dependent on the Web interface, Returns a Json String with the information
    '''
    response = {}
    response['teamname'] = name

    k = 5
    query_result = pvi.execute_query(name, k)

    for k, v in query_result.items():
        response[k] = v

    return response

@app.route('/report')
def report():
    # Get all team names associated with 2023 teams
    names_2023 = list(pvi.master_df.query("YEAR == 2023")['TEAM'])

    # Run a direct query for each 2023 team
    predictions = []
    for name in names_2023:
        predictions.append(direct_query(name))

    # Return a JSON list of all 2023 predictions
    return predictions

if __name__ == '__main__':
    app.run(debug=True)