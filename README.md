A test suite for full model FDR and power comparing
selective (conditional) inference and knockoffs.

To install
----------

git submodule update --init
cd R-software
git submodule update --init
make install
cd ../selective-inference
git submodule update --init
pip install -e .

Testing
-------

If installed correctly, the following should give you some results for a small problem:

python compare.py --instance indep_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html