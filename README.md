# trcrp_final
Final Version of TRCRP with supporting documentation and updated code. This is updated code from the following paper:  
['Temporally-Reweighted Chinese Restaurant Process Mixtures for Clustering, Imputing, and Forecasting Multivariate Time Series'](https://paperswithcode.com/paper/temporally-reweighted-chinese-restaurant)

## Setup

1. **Set up `crosscat`**:
    ```sh
    cd crosscat
    python setup.py build_ext --inplace
    pip install -e .
    python setup.py develop
    ```

2. **Set up `cgpm`**:
    ```sh
    cd cgpm
    pip install -e .
    python setup.py develop
    ```

3. **Set up `trcrpm`**:
    ```sh
    cd trcrpm
    pip install -e .
    python setup.py develop
    ```

**Experiments with supporting functions are found under the `experiments` folder.**
