from flask import Flask, render_template
from flask.globals import request
import pickle
filename = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('message')
        print(message)
        if not message:
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = classifier.predict(vect)
        else:
            my_prediction = -1
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)


