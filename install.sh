#!/bin/bash

set -e

here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ $# -eq 0 ]
then
  dest=$here
else
  dest=$1
fi

mkdir -p $dest
cd "$dest"
if [ -d flatbuffers ]
then
  git -C flatbuffers pull
else
  git clone https://github.com/google/flatbuffers.git
fi
cd flatbuffers
cmake .
make -j$(nproc)
cd -

flatbuffers/flatc --python --gen-object-api $here/schema/schema.fbs

# Nasty hack needed here as flatc does not allow you to control the package name
# the bindings are generated to and "tflite" is a package name in public pypi
# so a clash is highly likely.
mkdir -p tumeda_tflite/
mv tflite/* tumeda_tflite/
rm -rf tflite
cd tumeda_tflite
for f in *.py; do sed -e'1,$s/tflite\./tumeda_tflite./g' -i $f; done
cd -

mkdir -p $dest/tflite_pack
cp -r $here/tflite_pack/* $dest/tflite_pack

echo sed -e "s~%INSTALL_DIR%~$dest~g" $here/run.sh.template
sed -e "s~%INSTALL_DIR%~$dest~g" $here/run.sh.template > $dest/run.sh
chmod +x $dest/run.sh

cd $here
