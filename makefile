corpus:
	wget -P ./data/ "http://www.nilc.icmc.usp.br/nilc/tools/fapesp-corpora.tar.gz"
	gunzip ./data/fapesp-corpora.tar.gz
	gunzip ./data/fapesp-corpora/corpora/pt.tgz
	python prepare_corpus
