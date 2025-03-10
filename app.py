from flask import Flask, request, render_template
from joblib import load
import numpy as np

app = Flask(__name__)

model = load('decision_tree_model.joblib')
print("Expected Features:", model.n_features_in_)


# Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    result = None  # Default: No result initially
    form_data = None
    

    if request.method == 'POST':

        form_data = request.form

        input_data  = [
            int(request.form['bp']),
            int(request.form['chol']),               
            int(request.form['cholcheck']),
            int(request.form['bmi']),
            int(request.form['smoker']),
            int(request.form['stroke']),
            int(request.form['heartDiseaseOrAttack']), 
            int(request.form['physActivity']),
            int(request.form['fruits']),            
            int(request.form['veggies']), 
            int(request.form['hvyAlcoholConsump']),  
            int(request.form['anyHealthcare']),     
            int(request.form['noDocbcCost']),
            int(request.form['generalHealth']), 
            int(request.form['mentalHealth']),
            int(request.form['physicalHealth']),
            int(request.form['diffWalk']),
            int(request.form['gender']),        
            int(request.form['age']),   
            int(request.form['edu']),             
            int(request.form['income']),
            0,  
            0, 
            0             
        ]

        

        # Convert to NumPy array and reshape for model input
        input_data = np.array(input_data).reshape(1, -1)

        # Predict using the loaded model
        prediction = model.predict(input_data)[0]
        result = "High risk of diabetes" if prediction == 1 else "Low risk of diabetes"

    # Render the test page with the result (if any)
    return render_template('test.html', result=result, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)