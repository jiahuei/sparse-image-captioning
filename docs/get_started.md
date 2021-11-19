# Get Started

## Requirements

The main requirements for `sparse_caption` are as follows:

* python
* pytorch
* sentencepiece
* torchvision
* Java 8

The full list of required packages can be found `requirements_base.txt` and `requirements.txt`.

```{admonition} Dependency Versions
The exact dependency versions that are tested can be found in `requirements_dev_base.txt` and `requirements_dev.txt`.
```


## Installation and Setup

### Docker
Setup using Docker is the most convenient.

1. Run `docker build -t <YOUR_TAG> -f Dockerfile .` to build the image.
2. Run the command below to launch a container. **Change `<YOUR_DATASET_PATH>` to your desired path.** All dataset files will be placed inside `<YOUR_DATASET_PATH>`.
   ```bash
    docker run -it \
        --gpus all \
        --ipc=host \
        -v .:/workspace \
        -v <YOUR_DATASET_PATH>:/datasets \
        -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY \
        --rm <YOUR_TAG>
   ```
3. In the container, run `bash /workspace/scripts/dataset/setup.sh` to perform dataset download, pre-processing, and feature extraction.
4. Done.

```{warning}
Upon container launch, the `sparse_caption` package will be installed in editable mode.

Thus, the repository directory should be mounted to `/workspace` in the container.
```

### Pip

1. Install the package: `pip install -e .`
2. Install Java 8.
3. Done.

### Anaconda

```{admonition} Windows
1. On Windows, you might need to install "Microsoft C++ Build Tools" in order to build `pycocotools`, 
by [downloading from this link](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

2. Tick the options as shown:
    ![win-build-tools](win-build-tools.png)
```

1. These commands should install the required packages:
    ```bash
    conda install h5py matplotlib pandas Pillow scipy seaborn tqdm
    # conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    conda install -c pytorch torchtext
    conda install -c fastai sentencepiece
    conda install -c conda-forge tensorboard six

    # For dev, optional
    conda install pytest
    conda install -c conda-forge black doc8 myst-parser sphinx sphinx-autobuild sphinx_rtd_theme
    ```
2. Install the package: `pip install -e .`
3. Install Java 8.
4. Done.

```{admonition} Possible *pycocotools* installation issues
* One might run into issues related to `numpy` or `cython` when importing `pycocotools`.
* To resolve it, either:
    * Install `numpy` and `cython` prior to installing `pycocotools`
    * Maybe try using `pycocotools-fix` instead
```
