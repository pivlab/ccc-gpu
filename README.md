# Clustermatch Correlation Coefficient GPU (CCC-GPU)

## Development environment setup
At the root of the repository, run:
```
conda env create -f environment/environment-gpu.yml
```

Then, you can use the following script to activate the conda environment and set up PYTHONPATH and other configurations for the current shell session:
```
source ./scripts/setup_dev.sh
```

This script can also be configured as a startup script in PyCharm so you don't have to run it manually every time.

## Installation
Now the package can only be installed from source. At the root of the repository, run:
```
pip install -e .
```

Then you can import the package in your Python scripts.

## Documentation
Currently, this repository is not publicly accessible, making tools like ReadTheDocs not able to scan the codebase to build and publish the documentation. Thus, the documentation needs to be built locally:

```
cd docs
make html
```

Then you can access the html file `docs/build/html/index.html` to view the documentation.

If you are using VsCode, it's convenient to use the `Live Preview` extension to view the webpage within the code editor.
