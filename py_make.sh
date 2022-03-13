#!/bin/bash
# python -m pip uninstall -y faiss
cd /home/gauenk/Documents/packages/faiss/build/faiss/python
python -m pip install -e . --user
cd /home/gauenk/Documents/packages/faiss/
