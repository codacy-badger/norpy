# ROADMAP

We now want to work ahead and use norpy as soon as possible in as a replacement for the crude
RESPY adjustments in the SMM and MLE estimation and then add the additional state variable there. We use SME and MLE wrappers to estimate the model.

## Next projects

* We want to clean up the FORTRAN codes a little mode by aligning the indices (k, period) with the standard indexing in FORTRAN starting at 1 and not in zero. In addition we want to finish our work on the model specification type. This project does only affect the FORTRAN files.

* We need to clean up the property tests to follow the expected structure and increase test coverage.

* We need to polish the PYTHON codes by addressing the flake8 complaint and set up black as a pre-commit hock.
