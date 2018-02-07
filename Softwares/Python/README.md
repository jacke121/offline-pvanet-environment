# Python 离线安装

解压三个安装包

## 安装python

1. 首先进入Python目录，`./configure --prefix=/usr/local/python2.7`
2. `make -j`
3. `sudo make install`

## 安装setuptools

1. 进入setuptools目录，`sudo python setup.py build`
2. `sudo python setup.py install`

## 安装Anaconda

1. `sudo sh Anaconda.....sh`

## 安装opencv

### 编译安装opencv

1. 解压opencv2.4.13

2. 编译OpenCV，使用如下命令：

   ```
   cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D PYTHON_EXECUTABLE=/home/cuizhou/anaconda2/bin/python -D BUILD_OPENCV_PYTHON2=ON -D PYTHON_INCLUDE_DIR=/home/cuizhou/anaconda2/include/python2.7 -D PYTHON_LIBRARY=/home/cuizhou/anaconda2/lib/libpython2.7.so -D PYTHON_NUMPY_PATH=/home/cuizhou/anaconda2/lib/python2.7/site-packages -D PYTHON_PACKAGES_PATH=/home/cuizhou/anaconda2/lib/python2.7/site-packages ..
   make -j3
   ```

## 安装easydict

1. 进入easydict 目录
2. `python setup.py build`
3. `python setup.py intall`
4. 在运行的脚本里加入：`import sys`  `sys.path.insert(0, '$easydict 目录')`

## 安装protobuf的python支持

1. 进入protobuf/python
2. `python setup.py build`
3. 在运行的脚本里加入：`import sys`  `sys.path.insert(0, 'protobuf/python')`

