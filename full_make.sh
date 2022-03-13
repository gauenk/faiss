#!/bin/bash
# cmake -B build .
make -C build -j swigfaiss
python -m pip uninstall -y faiss
cd /home/gauenk/Documents/packages/faiss/build/faiss/python
python -m pip install -e . --user
# cd /home/gauenk/.local/lib/python3.8/site-packages
# unzip -o faiss-1.7.2-py3.8.egg
cd /home/gauenk/Documents/packages/faiss/
