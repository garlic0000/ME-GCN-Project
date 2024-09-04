cmake \
    -DBUILD_EXAMPLES=OFF \  #不编译 OpenCV 的示例代码。这有助于减少编译时间和二进制文件大小。

    -DWITH_QT=OFF \  #不启用与 Qt 库的集成。Qt 通常用于 GUI 应用程序，但如果不需要，可以禁用。

    -DCUDA_GENERATION=Auto \  #自动检测并配置 CUDA 代码生成器以支持 GPU 的架构。CUDA 代码生成器会为特定的 GPU 架构优化编译输出。

    -DOpenGL_GL_PREFERENCE=GLVND \  #指定 OpenGL 的实现。GLVND（OpenGL Vendor-Neutral Dispatch）是一种用于支持多个 OpenGL 实现的机制。

    -DBUILD_opencv_hdf=OFF \  #不编译 OpenCV 的 HDF 模块。HDF 模块提供了 HDF5 格式的数据读写功能，如果不需要，可以禁用。

    -DBUILD_PERF_TESTS=OFF \  #不编译性能测试模块，这样可以加快编译过程并减少最终构建的大小。

    -DBUILD_TESTS=OFF \  #不编译单元测试代码，从而减少编译时间。

    -DCMAKE_BUILD_TYPE=RELEASE \  #配置为 Release 构建类型，进行优化以提高性能。

    -DBUILD_opencv_cnn_3dobj=OFF \  #不编译 OpenCV 的 CNN 3D 对象识别模块。

    -DBUILD_opencv_dnn=OFF \  #不编译 OpenCV 的 DNN（深度神经网络）模块。如果不需要深度学习支持，可以禁用。

    -DBUILD_opencv_datasets=OFF \  #不编译数据集模块，该模块包含多个图像数据集的接口。

    -DBUILD_opencv_aruco=OFF \  #不编译 ArUco 模块，该模块用于检测 ArUco 标记（用于计算机视觉中的姿态估计）。

    -DBUILD_opencv_tracking=OFF \  #不编译跟踪模块，该模块用于跟踪物体在视频中的运动。

    -DBUILD_opencv_text=OFF \  #不编译文本模块，该模块用于文本检测和识别。

    -DBUILD_opencv_stereo=OFF \  #不编译立体视觉模块。

    -DBUILD_opencv_saliency=OFF \  #不编译显著性检测模块，用于检测图像中的显著区域。

    -DBUILD_opencv_rgbd=OFF \  #不编译 RGB-D 模块，用于处理 RGB 和深度数据。

    -DBUILD_opencv_reg=OFF \  #不编译图像配准模块。

    -DBUILD_opencv_ovis=OFF \  #不编译 OVIS 模块，该模块依赖于 OGRE3D，用于视觉仿真。

    -DBUILD_opencv_matlab=OFF \  #不编译 MATLAB 接口模块。

    -DBUILD_opencv_freetype=OFF \  #不编译 FreeType 模块，该模块用于字体渲染。

    -DBUILD_opencv_dpm=OFF \  #不编译 DPM 模块（Deformable Parts Model）。

    -DBUILD_opencv_face=OFF \  #不编译人脸识别模块。

    -DBUILD_opencv_dnn_superres=OFF \  #不编译用于超分辨率的深度神经网络模块。

    -DBUILD_opencv_dnn_objdetect=OFF \  #不编译用于对象检测的深度神经网络模块。

    -DBUILD_opencv_bgsegm=OFF \  #不编译背景分割模块。

    -DBUILD_opencv_cvv=OFF \  #不编译 OpenCV Visual Debugger（cvv），用于调试计算机视觉应用。

    -DBUILD_opencv_ccalib=OFF \  #不编译相机校准工具。

    -DBUILD_opencv_bioinspired=OFF \  #不编译生物启发的视觉系统模块。

    -DBUILD_opencv_dnn_modern=OFF \  #不编译现代深度神经网络模块。

    -DBUILD_opencv_dnns_easily_fooled=OFF \  #不编译易受愚弄的神经网络模块。

    -DBUILD_JAVA=OFF \  #不编译 Java 绑定。

    -DBUILD_opencv_python2=OFF \  #不编译 Python 2 绑定。

    -DBUILD_NEW_PYTHON_SUPPORT=ON \  #启用新的 Python 支持。

    -DBUILD_opencv_python3=OFF \  #不编译 Python 3 绑定。

    -DHAVE_opencv_python3=OFF \  #禁用 Python 3 支持。

    -DPYTHON_DEFAULT_EXECUTABLE="$(which python)" \  #指定默认的 Python 可执行文件路径。

    -DWITH_OPENGL=ON \  #启用 OpenGL 支持，用于硬件加速渲染。

    -DWITH_VTK=OFF \  #不启用 VTK 支持（Visualization Toolkit）。

    -DFORCE_VTK=OFF \  #不强制使用 VTK。

    -DWITH_TBB=ON \  #启用 TBB（Threading Building Blocks）支持，用于多线程优化。

    -DWITH_GDAL=ON \  #启用 GDAL（Geospatial Data Abstraction Library）支持，用于处理地理空间数据。

    -DCUDA_FAST_MATH=ON \  #启用 CUDA 快速数学功能，以优化 GPU 上的数学运算。

    -DWITH_CUBLAS=ON \  #启用 cuBLAS 支持，用于加速线性代数计算。

    -DWITH_MKL=ON \  #启用 Intel MKL（Math Kernel Library）支持，以优化数学运算。

    -DMKL_USE_MULTITHREAD=ON \  #启用 MKL 多线程支持。

    -DOPENCV_ENABLE_NONFREE=ON \  #启用非自由模块的编译，如 SIFT（尺度不变特征变换）。

    -DWITH_CUDA=ON \  #启用 CUDA 支持，用于 GPU 加速。

    -DNVCC_FLAGS_EXTRA="--default-stream per-thread" \  #为 NVCC 编译器设置额外的编译标志，指定 CUDA 的默认流为每线程独立流。

    -DWITH_NVCUVID=OFF \  #不启用 NVidia CUDA 视频解码器支持。

    -DBUILD_opencv_cudacodec=OFF \  #不编译 CUDA 视频编解码模块。

    -DMKL_WITH_TBB=ON \  #启用与 TBB 的 MKL 集成。

    -DWITH_FFMPEG=ON \  #启用 FFmpeg 支持，用于处理视频输入输出。

    -DMKL_WITH_OPENMP=ON \  #启用 MKL 与 OpenMP 的集成，以支持多线程。

    -DWITH_XINE=ON \  #启用 Xine 库的支持，用于视频播放。

    -DENABLE_PRECOMPILED_HEADERS=OFF \  #禁用预编译头，以减少编译依赖。

    -DCMAKE_INSTALL_PREFIX=/usr/local \  #指定安装路径为 /usr/local。

    -DOPENCV_GENERATE_PKGCONFIG=ON \  #生成 .pc 文件，用于 pkg-config 工具检测 OpenCV。

    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \  #指定 OpenCV contrib 模块的路径，用于扩展功能。

    .. \  #指定上级目录为源代码路径，即 CMakeLists.txt 文件所在的目录。