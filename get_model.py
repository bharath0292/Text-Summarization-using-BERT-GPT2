from transformers import GPT2Model,GPT2Tokenizer,BertModel,BertTokenizer

class fetch_model:

    def pretrained_model(self,model_name):

        if model_name=='bert':
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            bert_model = BertModel.from_pretrained('bert-base-uncased')

            bert_model.save_pretrained('models/bert-base-uncased')
            bert_tokenizer.save_pretrained('models/bert-base-uncased')

        elif model_name=='gpt2':
            gpt_tokenizer = GPT2Model.from_pretrained('gpt2')
            gpt_model = GPT2Tokenizer.from_pretrained('gpt2')

            gpt_model.save_pretrained('models/gpt2')
            gpt_tokenizer.save_pretrained('models/gpt2')
