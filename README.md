# offline-pvanet-environment

因为甲方要求部署环境时不能联网，而且服务器没有计算显卡，所以需要摸索在纯净ubuntu14.04.5中部署纯CPU的pvanet环境



## caffe依赖环境安装

### 安装build-essential

* 进入`build-essential`目录，依次输入以下命令：

  ```
  sudo dpkg -i libstdc++-4.8-dev_4.8.4-2ubuntu1~14.04.3_amd64.deb
  sudo dpkg -i g++-4.8_4.8.4-2ubuntu1~14.04.3_amd64.deb
  sudo dpkg -i dpkg-dev_1.17.5ubuntu5.7_all.deb
  sudo dpkg -i g++_4%3a4.8.2-1ubuntu6_amd64.deb
  sudo dpkg -i build-essential_11.6ubuntu6_amd64.deb
  ```

### 安装cmake

* 进入`cmake`目录，依次输入以下命令：

  ```
  sudo dpkg -i cmake-data_2.8.12.2-0ubuntu3_all.deb
  sudo dpkg -i cmake_2.8.12.2-0ubuntu3_amd64.deb
  ```

### 安装boost

* 进入`boost`目录，依次进入`boost_basic  boost_filesystem_regex  boost_thread   boost_python`输入以下命令：

  ```
  sudo dpkg -i libboost1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-dev_1.54.0.1ubuntu1_amd64.deb
  sudo dpkg -i libboost-system1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-system-dev_1.54.0.1ubuntu1_amd64.deb

  sudo dpkg -i libboost-filesystem1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-filesystem1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-filesystem-dev_1.54.0.1ubuntu1_amd64.deb
  sudo dpkg -i icu-devtools_52.1-3ubuntu0.7_amd64.deb
  sudo dpkg -i libicu52_52.1-3ubuntu0.7_amd64.deb
  sudo dpkg -i libicu-dev_52.1-3ubuntu0.7_amd64.deb
  sudo dpkg -i libboost-regex1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-regex1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-regex-dev_1.54.0.1ubuntu1_amd64.deb

  sudo dpkg -i libboost-atomic1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-atomic1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-chrono1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-chrono1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-serialization1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-serialization1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-date-time1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-thread1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-thread1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-thread-dev_1.54.0.1ubuntu1_amd64.deb

  sudo dpkg -i libboost-python1.54.0_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libexpat1_2.1.0-4ubuntu1.4_amd64.deb
  sudo dpkg -i libexpat1-dev_2.1.0-4ubuntu1.4_amd64.deb
  sudo dpkg -i libpython2.7-minimal_2.7.6-8ubuntu0.4_amd64.deb
  sudo dpkg -i libpython2.7-stdlib_2.7.6-8ubuntu0.4_amd64.deb
  sudo dpkg -i libpython2.7_2.7.6-8ubuntu0.4_amd64.deb
  sudo dpkg -i libpython2.7-dev_2.7.6-8ubuntu0.4_amd64.deb
  sudo dpkg -i libpython-dev_2.7.5-5ubuntu3_amd64.deb
  sudo dpkg -i libboost-python1.54-dev_1.54.0-4ubuntu3.1_amd64.deb
  sudo dpkg -i libboost-python-dev_1.54.0.1ubuntu1_amd64.deb
  ```

### 安装protobuf、atlas、hdf5、glog

* 同上



## 安装Python环境

### 安装python

1. 首先进入Python目录，`./configure --prefix=/usr/local/python2.7`
2. `make -j`
3. `sudo make install`

### 安装setuptools

1. 进入setuptools目录，`sudo python setup.py build`
2. `sudo python setup.py install`

### 安装Anaconda

1. `sudo sh Anaconda.....sh`

### 安装opencv

1. 解压opencv2.4.13

2. 编译OpenCV，使用如下命令：

   ```
   cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D PYTHON_EXECUTABLE=/home/cuizhou/anaconda2/bin/python -D BUILD_OPENCV_PYTHON2=ON -D PYTHON_INCLUDE_DIR=/home/cuizhou/anaconda2/include/python2.7 -D PYTHON_LIBRARY=/home/cuizhou/anaconda2/lib/libpython2.7.so -D PYTHON_NUMPY_PATH=/home/cuizhou/anaconda2/lib/python2.7/site-packages -D PYTHON_PACKAGES_PATH=/home/cuizhou/anaconda2/lib/python2.7/site-packages ..
   make -j3
   sudo make install
   ```

### 安装easydict

1. 进入easydict 目录
2. `python setup.py build`
3. `python setup.py intall`
4. 在运行的脚本里加入：`import sys`  `sys.path.insert(0, '$easydict 目录')`

### 安装protobuf的python支持

1. 进入protobuf/python
2. `python setup.py build`
3. 在运行的脚本里加入：`import sys`  `sys.path.insert(0, 'protobuf/python')`



## 编译caffe

* `caffe-for-pvanet`是配置好的用anaconda环境，纯cpu编译的caffe版本，相应更改可自行设置

1. 解压`caffe-for-pvanet.tar.gz`
2. `make all -j`
3. `make pycaffe -j`
4. 将`python`目录下编译好的`_caffe.so`拷贝至`PvaNet/recognition/distribute/python/caffe`目录下
5. 将`build/lib`目录下的文件拷贝至`PvaNet/recognition/distribute/build/lib`目录下



## 编译faster-rcnn相关lib

* 进入`PvaNet/recognition/distribute/lib`目录下，运行`make -j`即可



至此，ubuntu14.04.5下的纯cup  pvanet已配置完成