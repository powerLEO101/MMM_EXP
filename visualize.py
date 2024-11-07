from transformers import AutoTokenizer

tokenizer = None

def set_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def visualize(tokens):
    if len(tokens.shape) == 1:
        text = tokenizer.decode(tokens)
        print(text)
    else:
        text = tokenizer.batch_decode(tokens)
        print('\n'.join(text))
