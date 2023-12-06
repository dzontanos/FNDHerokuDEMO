from flask import Flask,render_template,url_for,request
from transformers import TextClassificationPipeline, AutoTokenizer, AutoModelForSequenceClassification

name = 'ZodiUOA/Covid19-fake-news'

tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained('ZodiUOA/Covid19-fake-news',  max_position_embeddings=512)

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

application = app = Flask(__name__)

@application.route('/')
def home():
	return render_template('home.html')
@application.route('/predict',methods=['POST'])

def predict():
    if request.method == 'POST':
        input_message = request.form['message']
        if len(input_message)>=511:
             input_message= input_message[0:512]
        if input_message.strip() == "":
            result="Please enter the body of an article"
        my_input = [input_message]
        preds = pipe(my_input, return_all_scores=True)
        output_dict = {'Real': preds[0][0]['score'], 'Fake': preds[0][1]['score']}
        print(output_dict)
        print(list(output_dict.keys()), list(output_dict.values()))
        props = [(round(float(v)*100, 2)) for v in list(output_dict.values())]
        print(props)
        return render_template('result.html', mess = input_message, classes = list(output_dict.keys()), props=props)

if __name__ == '__main__':
    app.run(port=5000,debug=True)
