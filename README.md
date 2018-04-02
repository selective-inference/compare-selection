A test suite for full model FDR and power comparing
selective (conditional) inference and knockoffs.

To install
----------

```
git submodule update --init;
cd R-selective;
git submodule update --init;
make install;
cd ../selective-inference;
git submodule update --init;
pip install -r requirements.txt;
pip install -e .;
cd ..;
```

Testing
-------

If installed correctly, the following should give you some results for a small problem:

python compare.py --instance AR_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile AR.html --csvfile AR.csv

For a grid of signal and / or AR(1) parameters try

python compare.py --instance AR_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile AR.html --signal 3 4 5 --rho 0 0.5 --csvfile AR.csv

Also, try 

python compare.py --help
python compare.py --list_instances
python compare.py --list_methods
 
##### Travis test
