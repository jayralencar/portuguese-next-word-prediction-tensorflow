corpus:
	wget -P ./data/ "http://www.nilc.icmc.usp.br/nilc/tools/fapesp-corpora.tar.gz"
	tar -xvzf  ./data/fapesp-corpora.tar -C data
	tar -xvzf ./data/fapesp-corpora/corpora/pt.tgz -C data/fapesp-corpora/corpora/
	python prepare_corpus.py
