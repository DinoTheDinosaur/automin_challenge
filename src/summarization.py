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



def create_dataloader(args, texts, tokenizer, device='cpu', max_pos=512, max_n_words=510):
    sep_vid = tokenizer.vocab['[SEP]']
    cls_vid = tokenizer.vocab['[CLS]']
    n_lines = len(open(source_fp).read().split('\n'))

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace('[cls]','[CLS]').replace('[sep]','[SEP]')
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0

        return src, mask_src, segments_ids, clss, mask_cls

    for text_id, text in enumerate(texts):
        parts = [[]]
        sentences = text.split('\n')
        cur_part_len = 0

        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            if (cur_part_len + len(words)) <= max_n_words:
                parts[-1] += [sentence]
                cur_part_len += len(words)
            else:
                parts += [[sentence]]
                cur_part_len = len(words)

        res = ''
        for part in parts:
            text_part = '\n'.join(part)
            src, mask_src, segments_ids, clss, mask_cls = _process_src(text_part)
            segs = torch.tensor(segments_ids)[None, :].to(device)
            batch = Batch()
            batch.text_id = text_id
            batch.src = src
            batch.tgt = None
            batch.mask_src = mask_src
            batch.mask_tgt = None
            batch.segs = segs
            batch.src_str = [[
                sent.replace('[SEP]','').strip() for sent in x.split('[CLS]')
            ]]
            batch.tgt_str = ['']
            batch.clss = clss
            batch.mask_cls = mask_cls
            batch.batch_size = 1
            yield batch


def predict_presumm(texts, model_path='models/cnndm_baseline_best.pt', tmp_dir='tmp/'):
    import nltk
    import torch

    from pytorch_transformers import BertTokenizer
    from src.models import AbsSummarizer, Batch, Translator

    checkpoint = torch.load(
        model_path
    )

    model = AbsSummarizer(
        checkpoint
    )
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True, cache_dir=tmp_dir
    )

    dataloader = create_dataloader(texts, tokenizer)

    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    predictor = Translator(model, tokenizer, symbols)
    results = []
    cur_text_id = 0
    for text_id, result in predictor.translate(dataloader):
        if results == []:
            results = [result]
        elif cur_text_id == text_id:
            results[-1] = ['\n'.join([results[-1], result])]
        else:
            results += [result]
            cur_text_id = text_id
    return results



SUMMARIZATIONS = {
    'random': predict_summary,
    't5': predict_T5,
    'pegasus': predict_pegasus,
    'cluster': predict_clustering,
    'presumm': predict_presumm
}