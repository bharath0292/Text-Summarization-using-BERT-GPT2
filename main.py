from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from summarizer import Summarizer,TransformerSummarizer
from transformers import GPT2Model,GPT2Tokenizer,GPT2Config,BertModel,BertTokenizer,BertConfig
import os
from get_model import fetch_model

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def convert_raw_text():

    data = request.json['data']
    min_length = request.json['min_length']
    max_length = request.json['max_length']
    model_name=request.json['model_name']

    fetch = fetch_model()

    if data=="":
        return jsonify({'summary': "Please give paragraph to summarize"})

    elif model_name.lower()=='gpt2':
        if os.path.isdir('models/gpt2'):
            pass
        else:
            fetch.pretrained_model('gpt2')

        custom_config = GPT2Config.from_pretrained('models/gpt2')
        custom_config.output_hidden_states = True
        custom_tokenizer = GPT2Tokenizer.from_pretrained('models/gpt2')
        custom_model = GPT2Model.from_pretrained('models/gpt2', config=custom_config)

        model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
        result = model(data, min_length=min_length, max_length=max_length)
        summary = "".join(result)

    elif model_name.lower()=='bert':

        if os.path.isdir('models/bert-base-uncased'):
            pass
        else:
            fetch.pretrained_model('bert')

        custom_config = BertConfig.from_pretrained('models/bert-base-uncased')
        custom_config.output_hidden_states = True
        custom_tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
        custom_model = BertModel.from_pretrained('models/bert-base-uncased', config=custom_config)

        model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
        result = model(data, min_length=min_length, max_length=max_length)
        summary = "".join(result)

    return jsonify({'model': model_name ,'summary': summary})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000,debug=True)
