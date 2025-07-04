#!/bin/bash
set -e

GIT_ROOT_DIR=$(git rev-parse --show-toplevel)

# This script runs a Jupyter notebook (.ipynb) from the command line using
# papermill.
#
# This script must be run within the nbs/ folder.

if [ -z "${1}" ]; then
    echo "Specify notebook to run"
    exit 1
fi

# If the notebook is an "output notebook" (*.run.ipynb), which are generated by
# papermill for instance, then do not run it.
pattern="*.run.ipynb"

input_notebook=$1
shift

if [[ $input_notebook == $pattern ]]; then
    echo "Not running output notebook"
    exit 0
fi

override_nbs=${CM_RUN_NBS_OVERRIDE}

# if second argument is a notebook, then it is the output
# notebook filename
if [[ $1 == *.ipynb ]]; then
    output_notebook=${input_notebook%/*}/$1
    shift

    # do not override if output was specified
    override_nbs=0
else
    output_notebook="${input_notebook%.*}.run.ipynb"
fi

# run papermill
papermill \
  --log-output \
  --request-save-on-cell-execute \
  $@ \
  $input_notebook \
  $output_notebook

# Convert to notebook
#
# This is to reduce the notebook final size, which is huge after
# running with papermill.
jupyter nbconvert --to notebook ${output_notebook} --output ${output_notebook##*/}

if [ "${override_nbs}" != "0" ]; then
    mv $output_notebook $input_notebook
    bash ${GIT_ROOT_DIR}/scripts/convert_ipynb_to_py.sh ${input_notebook}
fi
