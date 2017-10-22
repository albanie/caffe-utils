# set path to caffe dir
CAFFE_DIR=${1:-"~/coding/libs/crfasrnn/caffe-crfrnn"}
echo "symlinking debug layer source code into ${CAFFE_DIR}"

# add symlinks
ln -s ./src/caffe/debug_solver.cpp \
    "${CAFFE_DIR}/src/caffe/debug_solver.cpp"
ln -s ./include/caffe/debug_solver.hpp \
    "${CAFFE_DIR}/include/caffe/debug_solver.hpp"

# then recompile
