#!/bin/sh

cd docs

echo "[INFO] start scanning modules..."
if [ -d "generated" ]; then
  rm -rf generated
fi
sphinx-apidoc -o generated ../physiossl --force --module-first
make html
#make latexpdf
cd ..