import random

def predict_summary(texts):
    ### PLACEHOLDER ###
    result = []
    for text in texts:
        lines = text.split('\n')
        random_lines = random.sample(lines, 10)
        result += ['\n'.join(random_lines)]
    return result

def predict_T5(texts):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    result = []
    for text in texts:
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=5, early_stopping=True)
        # remove extra tokens from prediction
        pred = tokenizer.decode(outputs[0])[1:-1]
        result.append(tokenizer.decode(outputs[0]))

    return result

def predict_pegasus(texts, model=None, tokenizer=None):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    if not model:
       model_name = 'google/pegasus-large'
       tokenizer = PegasusTokenizer.from_pretrained(model_name)
       model = PegasusForConditionalGeneration.from_pretrained(model_name)

    result = []
    for text in texts:
       tokens = tokenizer.encode(text, truncation=True, return_tensors="pt")
       generated = model.generate(tokens)
       res = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
       result.append(res)

    return result
