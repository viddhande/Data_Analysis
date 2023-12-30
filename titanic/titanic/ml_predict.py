'''
  def prediction_model(pclass,sex,age,sibsp,parch,fare,embarked,title):
    import pickle
    x = [[pclass,sex,age,sibsp,parch,fare,embarked,title]]
    randomforest = pickle.load(open('Data_Science.ipynb','rb'))
    prediction = randomforest.predict(x)
    return prediction
    '''

def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import joblib

    # Load the model
    randomforest = joblib.load(open('Data_Science_model.joblib','rb'))

    # Prepare the input data
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]

    # Make predictions
    prediction = randomforest.predict(x)

    return prediction
