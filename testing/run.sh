#!/bin/bash

set -e

WD=$(pwd)
export QSPPRED_TEST_TUTORIAL=${QSPPRED_TEST_TUTORIAL:-true}
export QSPPRED_TEST_EXTRAS=${QSPPRED_TEST_EXTRAS:-true}
echo "Running tests with:"
echo "QSPPRED_TEST_TUTORIAL=$QSPPRED_TEST_TUTORIAL"
echo "QSPPRED_TEST_EXTRAS=$QSPPRED_TEST_EXTRAS"

cd test_pytest && ./run.sh && cd "$WD"
cd test_cli && ./run.sh && cd "$WD"
cd test_tutorial && ./prepare.sh && cd "$WD"
cd test_consistency && ./run.sh && cd "$WD"
if [ "$QSPPRED_TEST_TUTORIAL" == "true" ]; then
  cd test_tutorial && ./run.sh && cd "$WD"
fi