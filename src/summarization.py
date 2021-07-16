STOPWORDS =[
    'yeah', 'bye', 'hi', 'hello',
    'ehm', 'i', 'think', 'uh', 'actually',
    'i\'m', 'um', 'would', 'you', 'i\'ll',
    'we', 'yes', 'no', 'if', 'me', 'yep',
    'it\'s', 'okay', 'maybe', 'surely',
    'sure', 'ha', 'exactly', 'mrs', 'miss', 'mr'
]
BLOCKWORDS = [
    'link', 'zoom'
]

def predict_summary(texts):
    ### PLACEHOLDER ###
    import random
    result = []
    for text in texts:
        lines = text.split('\n')
        random_lines = random.sample(lines, 10)
        result += ['\n'.join(random_lines)]
    return result


def preprocess_text(text):
    clean_lines = []
    for line in text.split('\n'):
        line_cleaned = []
        for word in line.split():
            if word in BLOCKWORDS:
                break
            elif word in STOPWORDS:
                continue
            line_cleaned += [word]
        if len(line_cleaned) > 5:
            clean_lines += [' '.join(line_cleaned)]
    new_text = '\n'.join(clean_lines)
    return new_text


def predict_T5(texts):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    result = []
    for text in texts:
        text = '.\n'.join(text.split('\n'))
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512)
        outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=5, early_stopping=True)
        # remove extra tokens from prediction
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result.append(pred)

    return result


def predict_pegasus(texts, model=None, tokenizer=None):
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    if not model:
        model_name = 'google/pegasus-large'
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

    result = []
    for text in texts:
        text = '.\n'.join(text.split('\n'))
        tokens = tokenizer.encode(text, truncation=True, return_tensors="pt")
        generated = model.generate(tokens)
        res = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
        result.append(res)

    return result

def predict_clustering(texts):
    import spacy
    import numpy as np
    import tensorflow_hub as hub
    import tensorflow.compat.v1 as tf
    from sklearn.cluster import AffinityPropagation, KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances
    tf.disable_eager_execution()
    nlp = spacy.load('en_core_web_sm')
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    new_texts = []
    for text in texts:
        new_phrases = ['']
        for sentence in text.split('\n'):
            doc = nlp(sentence)
            for token in doc:
                parent = token.head
                if token.dep_ == 'cc':
                    new_phrases += ['']
                elif (token.dep_ not in ['advmod', 'npadvmod', 'intj']) and (token.pos_ not in ['DET', 'INTJ', 'SCONJ']):
                    new_phrases[-1] += f' {token.text}'
        new_new_phrases = ['']
        for new_phrase in new_phrases:
            new_phrase = new_phrase.strip()
            if new_phrase == '':
                continue
            sentences = list(nlp(new_phrase).sents)
            for token in sentences[0]:
                if token.dep_ == 'nsubj':
                    new_new_phrases += [token.text]
                if token.dep_ != 'nsubj':
                    new_new_phrases[-1] += f' {token.text}'
        new_data = []
        for new_new_phrase in new_new_phrases:
            words = new_new_phrase.strip().split()
            new_words = [v for i, v in enumerate(words) if i == 0 or v != words[i-1]]
            new_data += [' '.join(new_words)]
        vectorizer = TfidfVectorizer(ngram_range=(1,3))
        X = vectorizer.fit_transform(new_data)
        importances = np.squeeze(np.asarray(X.sum(axis=1)))
        importances_n_texts = list(zip(
            importances,
            new_data
        ))
        embeddings = embed(new_data)
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(embeddings)
        clustering = AffinityPropagation(
            damping=0.9,
            preference=-1,
            convergence_iter=50,
            max_iter=1000
        ).fit(message_embeddings)
        print(len(set(clustering.labels_)))
        if -1 in clustering.labels_:
            clustering = KMeans(
                n_clusters=100, max_iter=1000, n_init=50
            ).fit(message_embeddings)
        cluster_importances = []
        for label in set(clustering.labels_):
            cluster_importance = np.mean(
                [importance for i, importance in enumerate(importances) if clustering.labels_[i] == label]
            )
            cluster_importances += [(
                label,
                cluster_importance
            )]
        importances_cluster_mapping = dict(cluster_importances)
        cluster_centers = {}

        for sel_label, importance in sorted(cluster_importances, key=lambda x: x[1], reverse=True):
            text = '\n'.join(list([line for label, line in zip(
                clustering.labels_,
                new_data
            ) if label == sel_label]))
            embedding_distances = cosine_distances(
                [clustering.cluster_centers_[sel_label]],
                message_embeddings
            )[0]
            center_phrase = new_data[
                embedding_distances.argmin()
            ]
            cluster_centers[sel_label] = center_phrase
        threshold = np.quantile(list(importances_cluster_mapping.values()), 0.9)
        summary_lines = list([line for label, line in zip(
            clustering.labels_,
            new_data
        ) if (line in cluster_centers.values()) and (importances_cluster_mapping[label] > threshold)])
        new_texts += ['\n'.join(summary_lines)]
    return new_texts



SUMMARIZATIONS = {
    'random': predict_summary,
    't5': predict_T5,
    'pegasus': predict_pegasus,
    'cluster': predict_clustering
}