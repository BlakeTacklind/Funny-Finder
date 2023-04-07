from flask import Flask
from flask import render_template
from flask import request
from flask import Markup

from predictor.predictor import Predictor, TEST_MODEL, MODEL_1, MODEL_2

import math

app = Flask(__name__)

# predictor = Predictor(TEST_MODEL)
predictor = Predictor()
print("ready to start")

input_value = None
output = None

@app.route("/", methods=['GET', 'POST'])
def main_app():
    global output

    if request.method == 'POST':
        #eww, but i don't know flask that good
        global input_value
        global predictor
        
        #run a prediction
        if request.form.get('predict') == 'Predict':
            input_value = request.form.get("transcript_input")

            output = runPrediction()

        #reset input
        elif request.form.get('Reset_action') == 'Reset':
            input_value = None
            output = None
            predicted = None

        #change model request from ui
        elif request.form.get('change_model') == 'Change Model':
            changed = predictor.changeModel(int(request.form.get("model_select")))

            #change the prediction if we have a different one
            if changed and input_value:
                output = runPrediction()

        else:
            pass # unknown
    elif request.method == 'GET':
        pass

    return render_template('main.html', output=output, currentModelText=getCurrentModelText())

def runPrediction():
    global input_value

    predicted = predictor.predict(input_value)
    return predictionToHTML(predicted)

#just get a string to represent which model is currently loaded
def getCurrentModelText():
    global predictor
    return predictor.getModelDescription()

#new line characters aren't displayed well, replace them with break lines
def newLineToBR(text):
    return text.text_with_ws.replace('\n', '<br>')


#turn the prediction into something the html and javascript can render
def predictionToHTML(predictions):
    def wordToHTML(word):
        if math.isnan(word['value']):
            return f'<span style="background:red">{newLineToBR(word["word"])}</span>'

        return f'<span class="value_holder" value="{word["value"]}" style="background-color:white" >{newLineToBR(word["word"])}</span>'
        
    return Markup("".join(wordToHTML(res) for res in predictions))

def create_app():
   return app

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
