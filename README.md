# Clustermatch Correlation Coefficient GPU (CCC-GPU)

The Clustermatch Correlation Coefficient (CCC) is a highly-efficient, next-generation not-only-linear correlation coefficient that can work on numerical and categorical data types. This repository contains the code of CCC and instructions to install and use it. It also has all the scripts/notebooks to run the analyses associated with the manuscript, where we applied CCC on gene expression data.

CCC-GPU is a GPU-accelerated version of CCC that can work on large datasets. It now supports the CUDA backend.

## Documentation
Currently, this repository is not publicly accessible, making tools like ReadTheDocs not able to scan the codebase to build and publish the documentation. Thus, the documentation needs to be built and viewed locally:

```
cd docs
make html
```

Then you can access the html file `docs/build/html/index.html` to view the documentation.

If you are using VsCode, it's convenient to use the `Live Preview` extension to view the webpage within the code editor.
