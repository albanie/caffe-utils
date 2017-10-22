# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

REFRESH=${1:-false}

# set path to caffe dir
CAFFE_DIR=${2:-"${HOME}/coding/libs/caffes/official-caffe"}
echo "symlinking debug layer source code into ${CAFFE_DIR}"
echo  "-----------"

declare -a rel_paths=(
"src/caffe/debug_solver.cpp"
"include/caffe/debug_solver.hpp"
)

# refresh previous symlinks
for rel_path in "${rel_paths[@]}"
do
   src="${SCRIPTPATH}/${rel_path}"
   dest="${CAFFE_DIR}/${rel_path}"
   if [ $REFRESH = false ] ; then
     if [ -f $src ] ; then
       link=false
     else
       link=true
     fi
   else # refresh
     echo "refreshing $dest"
     link=true
     if [ -L $dest ] ; then
       echo "deleting $dest"
       rm $dest
     fi
   fi
   if [ $link = true ] ; then
     echo "creating symlink ${src} -> ${dest}"
     ln -s ${src} ${dest}
   else
     echo "${dest} exists, skipping... "
   fi 
done
