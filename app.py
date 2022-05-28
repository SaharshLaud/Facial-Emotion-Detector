from statistics import mode
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


# create route for index.html
@app.route('/')
def index():
    return render_template('index.html')
    
# create route for after.html after button click on index.html
@app.route('/after', methods=['GET', 'POST'])
def after():
    # store the uploaded image by user in static folder
    img = request.files['file1']
    img.save('static/file.jpg')


   
    
    img = img/255.0

    img = np.reshape(img, (1,48,48,1))


    # load the CNN model trained previously and make predictions
    model = load_model('emotion_detection_model.h5')
    prediction = model.predict(img)

    label_map =   ['Anger','Neutral' , 'Fear', 'Happy', 'Sad', 'Surprise']

    prediction = np.argmax(prediction)

    final_prediction = label_map[prediction]




    
    return render_template("after.html", data=final_prediction)

















if __name__ == "__main__":
    app.run(debug=True)
