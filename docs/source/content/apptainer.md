# Apptainer

All scripts and jupyter notebooks can be run from a container. This has the advantage that all code is running 
independently of the system and that all resources (exp. all system cores) can be used.

To start this container ```apptainer``` has to be installed on the underlying system
[[quick-installation-steps](https://apptainer.org/docs/user/main/quick_start.html#quick-installation-steps)].
```Apptainer``` is only running on Linux-based systems natively. If the system is a Mac or Windows ```VirtualBox``` 
[[VirtualBox](https://www.virtualbox.org/)] can be installed. So Linux can be run in a separate environment on a 
non-Linux system.

## build a .sif-file

Next a .sif-file (comparable with a Docker-File or better Docker-image) is needed to start the container. The 
```sosall.sif``` file has all packages defined to run all SOSALL-code.

If a new .sif-file should be build, this can be achieved with a .def-file 
(comparable with Docker-File)

``` text
Bootstrap: docker
From: python:3.9.13-buster

%setup
    mkdir ${APPTAINER_ROOTFS}/data && mkdir ${APPTAINER_ROOTFS}/src

%files
    requirements.txt /opt
    pkg pkg

%environment
     export WD=${APPTAINER_ROOTFS}/src
     export SYS=APPTAINER

%post
   NOW=`date`
   echo "export NOW=\"${NOW}\"" >> $APPTAINER_ENVIRONMENT
   apt-get update && apt-get install vim -y
   cd pkg && pip install sklearn_custom-0.0.23-py3-none-any.whl && pip install PredictorPipeline-0.0.18-py3-none-any.whl && pip install misc-0.0.11-py3-none-any.whl && pip install statTests-0.0.3-py3-none-any.whl
   cd ../opt && pip install -r requirements.txt
```

To generate the requirements.txt from a underlying conda-environment type:

```text
pip list --format=freeze > requirements.txt 
```

The own written packages are in folder pkg as .whl-files:
* misc-0.0.11-py3-none-any.whl
* PredictorPipeline-0.0.18-py3-none-any.whl
* sklearn_custom-0.0.23-py3-none-any.whl
* statTests-0.0.3-py3-none-any.whl

In requirements.txt there will also be the own written packages (for example
PredictorPipeline-0.0.18-py3-none-any.whl, statTests-0.0.3-py3-none-any.whl, ...). These
have to be removed from the requirements.txt because they are installed manually
from the pkg-folder in the .sif-file building process.

From this .def-File a .sif-file is build with:

```text
sudo apptainer build sosall.sif sosall.def
```

## working with the apptainer

It's possible to start the container in a shell-mode. In this shell-mode every part of the code can be executed as on a 
normal (Linux-based) system.

One advantage of ```apptainer``` is that some resources (folders) will be 
automatically mounted when the container is started. So if the file structure is:

* **data** (with data/input/ and data/output)
* **src** (all source code)
* **sosall.sif**

we have access to all resources needed when running code inside the container.

### start container in shell mode 

```text
apptainer shell sossall.sif
```

### run script in shell-mode

```text
python3 ml_models.py
```

### run jupyter notebook in shell-mode

```text
ipython
run statistical_tests.ipynb
```

