#!/bin/bash
set -v
export DASHROOT=`pwd`
export SHA=`git log --pretty=format:'%h' -n 1`
export PATH=.:/usr/local/bin:${PATH}
export PYTHONPATH=${TRAVIS_BUILD_DIR}/build/lib
export LD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export DYLD_LIBRARY_PATH=${TRAVIS_BUILD_DIR}/build/lib
export MPLBACKEND=Agg
export TMPDIR=/tmp
mkdir build
ctest -S ${DASHROOT}/test/travis_ci/ctest_osx.cmake -V --timeout 180
status=$?
if [ ${status} -ne 0 ]
then
    exit 1
else
    exit 0
fi
