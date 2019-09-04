#!/bin/bash
set -v
export DASHROOT=`pwd`
#export SHA=`git --git-dir=${TECA_DIR}/.git log --pretty=format:'%h' -n 1`
export SHA=`git log --pretty=format:'%h' -n 1`
if [[ "$DOCKER_IMAGE" == "fedora" ]]; then
    source /usr/share/Modules/init/bash
    module load mpi
fi
export PATH=.:${PATH}
export PYTHONPATH=${DASHROOT}/build/lib
export LD_LIBRARY_PATH=${DASHROOT}/build/lib
export MPLBACKEND=Agg
mkdir build
cmake --version

ctest -S ${DASHROOT}/test/travis_ci/docker/ctest_linux.cmake -V --timeout 180
status=$?
if [ ${status} -ne 0 ]
then
    echo "TESTS FAILED! $?" 
    exit 1
else
    echo "TESTS PASSED! $?" 
    exit 0
fi
