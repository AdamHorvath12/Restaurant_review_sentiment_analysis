from flask import Flask, render_template, request
import pickle

modelname = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(modelname, 'rb'))
cv = pickle.load('cv-transform.pkl','rb')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        result_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=result_prediction)
        
if __name__ == '__main__':
	app.run(debug=True)