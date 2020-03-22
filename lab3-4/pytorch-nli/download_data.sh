#!/bin/bash

# Create folder structure
echo -e "\nCreating the folder structure...\n"
mkdir .data
mkdir .data/multinli
mkdir .data/snli
mkdir .vector_cache
echo -e "Done!"

# Download and unzip GloVe word embeddings
echo -e "\nDownloading and unzipping Glove 840B 300D to .vector_cache\n"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip -a glove.840B.300d.zip -d .vector_cache/
rm -f glove.840B.300d.zip
echo -e "\nDone!"

# Download and unzip NLI corpora

# MultiNLI:
echo -e "\nDownloading and unzipping MultiNLI to .data\n"
wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
unzip -a multinli_1.0.zip -d .data/multinli/
rm -f multinli_1.0.zip
echo -e "\nDone!"
cp datasets/multinli_0.9_test_mismatched_unlabeled.jsonl .data/multinli/multinli_1.0/
cp datasets/multinli_0.9_test_matched_unlabeled.jsonl .data/multinli/multinli_1.0/

# SNLI
echo -e "\nDownloading and unzipping SNLI to .data\n"
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip -a snli_1.0.zip -d .data/snli/
rm -f snli_1.0.zip
echo -e "\nDone!"

echo -e "\nAll Done!"
