pip install -r requirements.txt
mkdir modules
cd modules
git clone https://github.com/xsuite/xmask.git
pip install -e xmask
cd xmask
git submodule init
git submodule update
cd ..
git clone https://github.com/PyCOMPLETE/FillingPatterns.git
pip install -e FillingPatterns
cd ..
xsuite-prebuild