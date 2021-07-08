from src.train_abstractive import test_text_abs
import argparse

import nltk

def predict_one_file(text):

    args = argparse.Namespace()
    args.test_from = '../models/cnndm_baseline_best.pt'
    args.visible_gpus = 0
    args.large = False
    args.temp_dir = '../temp'
    args.finetune_bert = False
    args.enc_dropout = 0.2
    args.max_pos = 512
    args.share_emb = False
    args.dec_heads = 8
    args.dec_dropout = 0.2
    args.text_src = '../test'
    args.text_tgt = ''
    args.alpha = 0.6
    args.beam_size = 5
    args.min_length = 15
    args.max_length = 150
    args.max_tgt_len = 140
    args.model_path = '../models/'
    args.result_path = '../results/cnndm'
    args.recall_eval = False
    args.block_trigram = True

    NUM_WORDS = 128
    parts = []
    sentences = nltk.word_tokenize(text)

    for i in range(len(sentences)//NUM_WORDS):
        if i != (len(sentences)//NUM_WORDS -1):
            part = ' '.join(sentences[i*NUM_WORDS : (i+1) * NUM_WORDS])
        else:
            part = ' '.join(sentences[i * NUM_WORDS: ])
        parts.append(part)
    num_original = len(sentences)
    num = num_original

    while num >= int(0.1 * num_original):
        print("NEW ITERATION")
        res = ''
        for part in parts:
            with open("../test", "w") as f:
                f.write("%s \n" % (part))
            minute = test_text_abs(args)
            print("ORIGINS: %s" % (part))
            print("HYPOTHESIS: %s" % (minute))
            res += ' ' + minute
        sentences = nltk.word_tokenize(res)
        parts = []
        print(len(sentences))
        for i in range(len(sentences) // NUM_WORDS):
            if i != (len(sentences) // NUM_WORDS - 1):
                part = ' '.join(sentences[i * NUM_WORDS: (i + 1) * NUM_WORDS])
            else:
                part = ' '.join(sentences[i * NUM_WORDS:])
            parts.append(part)

        if not parts:
            break

        num = len(sentences)

    minute = ' '.join(sentences)
    print("RESULT: %s" % (minute))

    return minute

# TEST input text - return minute
# predict_one_file("And that's the availability of [ORGANIZATION4] for the upcoming [PROJECT1] call that's "
#                        "not related for what what the technical things that we want to discuss today at all.Uhm."
#                        "But I would like to remind [PERSON7] of this Doodle. So, [PERSON7] if you could fill uhm"
#                        " this Doodle poll, it's only selecting the day for now. Uh, but we need to announce the"
#                        " day two weeks had uhm so that we are uhm  we are according to the constitution agreement."
#                        " So that's why we need to choose the day now.(PERSON7) Yeah, you really uhm field my data."
#                        " Uhm.I will also can ask [PERSON2] to fill the to fill uhm the avalabiality the Doodle."
#                        " And my only concern is but I will not be in the office from 23rd to 2nd of May. To April "
#                        "23rd to May the second.So I only have the next week, not next week uhm the the following one."
#                        "(PERSON3) The weeks. And the the first of the two is, so fifteen -. "
#                        "(PERSON7) Fifteen, seventeen. (PERSON3) Fifteen and seventeen. Uhm. Yes, so hopefully we will "
#                        "be able to to find the date uhm. Yeah. Okay. So we will we will. (PERSON8) Currently it "
#                        "seems that uh, probably one of the best fitting would be April the sixteen. (PERSON3) "
#                        "Yes, so hopefully uhm yeah. If will work out that way, hopefully. Actually write it down "
#                        "in my callendar right away.Sixteen, that's Tuesday. So that should be uhm - No April, May."
#                        "Okay. Yeah. Uhm. Call. I'm already taking a note uhm in to my callendars. So hopefully this "
#                        "will work out.Thank you. Uhm, uhm. <another_yawn> So. Another thing is uhm that if you have"
#                        " any photos from the trade fair. That will be great. I sent you an email uhm. Some days ago."
#                        "Uhm and sh- in that email I'll send you -. Sorry. I just want to disable some stupid thing, "
#                        "how do I disable this? Uh-huh, hard crowed disk indeed never when I click uhm -. Where is "
#                        "remove? (PERSON7) Remove from Chrome? (PERSON3) Remove from Chrome. Yes. Remove. Okay. "
#                        "<another-language> Kdo to udělal? Okay. Yes. Sorry. I'm.Uhm. Yeah. I I lost the document "
#                        "for second, because it was uhm like uhm all the time showing some crazy overlay tool (dates), "
#                        "that (quality). Uhm, ehm, yeah, so uh, yeah. (PERSON7) We checked if you have in eh available "
#                        "for sharing the little <unintelligible> I have very <unintelligible>. (PERSON3) Yes. So we have "
#                        "uhm even very few could be useful as you know, we have already post, as you may know we have "
#                        "already post the news item on the [PROJECT1] website on this. And we are also putting together "
#                        "uhm a little video uhm on that session. And we have your recordings and colleague of mine"
#                        " is looking at those, and I've like put them in some order into a little screenplay for this."
#                        "So you are in, unless the technical quality proves to be too bad. But but it seems that uhm "
#                        "it it should work out, at least from what I have seen.Uh.")