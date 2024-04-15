from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    model_choice = request.form['model_choice']
    
    # Load the appropriate model based on the user's choice
    if model_choice == 'logistic_regression':
        model = pickle.load(open('logistic_regression_model.sav', 'rb'))
    elif model_choice == 'knn':
        model = pickle.load(open('knn_model.sav', 'rb'))
    elif model_choice == 'decision_tree':
        model = pickle.load(open('decision_tree_model.sav', 'rb'))
    
    # Make the prediction
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
