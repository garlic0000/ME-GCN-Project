#!/bin/bash
# install opencv
set -e

# ROOTDIR=${ZZROOT:-$HOME/app}
ROOTDIR=/usr/local
NAME1="opencv"
NAME2="opencv_contrib"
TYPE=".tar.gz"
FILE1="$NAME1$TYPE"
FILE2="$NAME2$TYPE"
# 更换链接
DOWNLOADURL1="https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz"
DOWNLOADURL2="https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.tar.gz"
echo $NAME1 will be installed in "$ROOTDIR"

mkdir -p "$ROOTDIR/downloads"
cd "$ROOTDIR"

if [ -f "downloads/$FILE1" ]; then
    echo "downloads/$FILE1 exist"
else
    echo "$FILE1 does not exist, downloading from $DOWNLOADURL1"
    wget $DOWNLOADURL1 -O $FILE1
    mv $FILE1 downloads/
fi

if [ -f "downloads/$FILE2" ]; then
    echo "downloads/$FILE2 exist"
else
    echo "$FILE2 does not exist, downloading from $DOWNLOADURL2"
    wget $DOWNLOADURL2 -O $FILE2
    mv $FILE2 downloads/
fi

mkdir -p src/$NAME1
mkdir -p src/$NAME2
tar xf downloads/$FILE1 -C src/$NAME1 --strip-components 1
tar xf downloads/$FILE2 -C src/$NAME2 --strip-components 1

cd src/$NAME1
mkdir -p build
cd build

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig/gtk+-3.0.pc:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig/gstreamer-1.0.pc:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig/gtkglext-1.0.pc:$PKG_CONFIG_PATH


cmake \
    -DBUILD_EXAMPLES=OFF \
    -DOPENCV_PYTHON3_VERSION=3.8 \
    -DPYTHON_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON_DEFAULT_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_EXECUTABLE=/opt/conda/bin/python3 \
    -DPYTHON3_INCLUDE_DIR=/opt/conda/envs/newCondaEnvironment/include/python3.8 \
    -DPYTHON3_LIBRARY=/opt/conda/envs/newCondaEnvironment/lib/libpython3.8.so \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages/numpy/core/include \
    -DPYTHON3_PACKAGES_PATH=/opt/conda/envs/newCondaEnvironment/lib/python3.8/site-packages \
    -DINSTALL_PYTHON_EXAMPLES=ON \
    -DINSTALL_C_EXAMPLES=OFF \
    -DWITH_QT=OFF \
    -DCUDA_GENERATION=Auto \
    #-DOpenGL_GL_PREFERENCE=GLVND \
    -DBUILD_opencv_hdf=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -DProtobuf_INCLUDE_DIR=/usr/include/google/protobuf/ \
    -DProtobuf_LIBRARIES=/usr/lib/x86_64-linux-gnu/libprotobuf.so \
    -DGLOG_INCLUDE_DIR=/usr/include/glog/ \
    -DGLOG_LIBRARY=/usr/lib/x86_64-linux-gnu/libglog.so \
    -DGFLAGS_INCLUDE_DIR=/usr/include/gflags/ \
    -DGFLAGS_LIBRARY=/usr/lib/x86_64-linux-gnu/libgflags.so \
    -DOGRE_DIR=/usr/include/OGRE/ \
    -DBUILD_opencv_cnn_3dobj=OFF \
    -DBUILD_opencv_dnn=ON \
    -DBUILD_opencv_datasets=OFF \
    -DBUILD_opencv_aruco=OFF \
    -DBUILD_opencv_tracking=OFF \
    -DBUILD_opencv_text=OFF \
    -DBUILD_opencv_stereo=OFF \
    -DBUILD_opencv_saliency=OFF \
    -DBUILD_opencv_rgbd=OFF \
    -DBUILD_opencv_reg=OFF \
    -DBUILD_opencv_ovis=OFF \
    -DBUILD_opencv_matlab=OFF \
    -DBUILD_opencv_freetype=OFF \
    -DBUILD_opencv_dpm=OFF \
    -DBUILD_opencv_face=OFF \
    -DBUILD_opencv_dnn_superres=OFF \
    -DBUILD_opencv_dnn_objdetect=OFF \
    -DBUILD_opencv_bgsegm=OFF \
    -DBUILD_opencv_cvv=OFF \
    -DBUILD_opencv_ccalib=OFF \
    -DBUILD_opencv_bioinspired=OFF \
    -DBUILD_opencv_dnn_modern=OFF \
    -DBUILD_opencv_dnns_easily_fooled=OFF \
    -DBUILD_JAVA=OFF \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_NEW_PYTHON_SUPPORT=ON \
    -DBUILD_opencv_python3=OFF \
    -DHAVE_opencv_python3=OFF \
    #-DWITH_OPENGL=ON \
    -DWITH_OPENGL=OFF \
    -DWITH_VTK=OFF \
    -DFORCE_VTK=OFF \
    -DWITH_TBB=ON \
    -DWITH_GDAL=ON \
    -DCUDA_FAST_MATH=ON \
    -DWITH_CUBLAS=ON \
    -DWITH_MKL=ON \
    -DMKL_USE_MULTITHREAD=ON \
    -DOPENCV_ENABLE_NONFREE=ON \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DOPENCV_DNN_CUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so \
    -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ \
    -DNVCC_FLAGS_EXTRA="--default-stream per-thread" \
    -DWITH_NVCUVID=OFF \
    -DBUILD_opencv_cudacodec=OFF \
    -DMKL_WITH_TBB=ON \
    -DWITH_FFMPEG=ON \
    -DMKL_WITH_OPENMP=ON \
    -DWITH_XINE=ON \
    -DWITH_AVRESAMPLE=OFF \
    -DGLIB_INCLUDE_DIR=/usr/include/glib-2.0 \
    -DGTK_INCLUDE_DIR=/usr/include/gtk-3.0 \
    -DGSTREAMER_INCLUDE_DIR=/usr/include/gstreamer-1.0 \
    -Ddc1394_DIR=/usr/include/libdc1394-2 \
    -DWITH_JULIA=ON \
    -DJULIA_EXECUTABLE=/usr/local/julia-1.7.3/bin/julia \
    -DJULIA_INCLUDE_DIR=/usr/local/julia-1.7.3/include/julia \
    -DJULIA_LIBRARIES=/usr/local/julia-1.7.3/lib/libjulia.so \
    -DVA_INCLUDE_DIR=/usr/include/va \
    -DOpenBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -DOpenBLAS_LIB=/usr/lib/x86_64-linux-gnu/libopenblas.so \
    -DAtlas_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas \
    -DAtlas_LIBRARIES=/usr/lib/x86_64-linux-gnu/libatlas.so \
    -DAtlas_CLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu/atlas \
    -DAtlas_CLAPACK_LIBRARY=/usr/lib/x86_64-linux-gnu/libatlas.so \
    -DOGRE_INCLUDE_DIR=/usr/include/OGRE \
    -DOGRE_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu/OGRE-1.9.0 \
    -DLAPACK_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
    -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/liblapack.so \
    -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libblas.so \
    -DOPENCV_DNN=ON \
    -DENABLE_PRECOMPILED_HEADERS=OFF \
    -DCMAKE_INCLUDE_PATH=/usr/include \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    ..

make -j"$(nproc)" && make install

echo "$NAME1" installed on "$ROOTDIR"
echo add following line to .zshrc
echo export OpenCV_DIR="$ROOTDIR"