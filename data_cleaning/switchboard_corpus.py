from convokit import Corpus, download

corpus = Corpus(filename=download("switchboard-corpus"))

utt = corpus.random_utterance()
print(utt)
print(utt.meta)

corpus.print_summary_stats()
