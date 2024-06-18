# trcrp_final
Final Version of TRCRP with supporting documentation and updated code. This is updated code from the following paper:
'Temporally-Reweighted Chinese Restaurant Process Mixtures for Clustering, Imputing, and Forecasting Multivariate Time Series'
https://paperswithcode.com/paper/temporally-reweighted-chinese-restaurant



To set up:

cd crosscat
python setup.py build_ext --inplace
pip install -e .
python setup.py develop


cd cgpm
pip install -e .
python setup.py develop

cd trcrpm
pip install -e .
python setup.py develop

**Experiments with supporting funtions are found under the experiments folder**
