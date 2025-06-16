#!/bin/bash
mkdir -p embeddings
wget -O embeddings/cc.he.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.he.300.vec.gz
gunzip embeddings/cc.he.300.vec.gz
wget -O embeddings/cc.en.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip embeddings/cc.en.300.vec.gz