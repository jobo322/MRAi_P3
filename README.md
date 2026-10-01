# MRAi_P3: Magnetic Resonance processing with Artificial intelligence for Peak Probability Presentarion

## Authors
* [Amir Jahangiri](amir.jahangiri@gu.se)
* [Vladislav Orekhov](vladislav.orekhov@nmr.gu.se)

## License
This package is licensed under the MIT License. See the LICENSE file for details.

## Citation
Please cite our articles if you use this application in your research or publication. 

* [NMR spectrum reconstruction as a pattern recognition problem](https://doi.org/10.1016/j.jmr.2022.107342)
* [Beyond traditional magnetic resonance processing with artificial intelligence](https://doi.org/10.1038/s42004-024-01325-w)
* [Towards ultimate nmr resolution with deep learning](https://arxiv.org/abs/2502.20793)

---

## Installation

1. Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) by following the Regular installation instructions.
2. Download MRAi_P3 pacckege from [`Code`](https://github.com/Amir-Jahangiri/MRAi_P3/archive/refs/heads/main.zip).
3. Unzip the `MRAi_P3-main.zip` file.
4. Open Terminal in `MRAi_P3-main` folder.
5. Create a conda environment with: 
```python
conda create -n mrai_p3 python=3.11
```
* Type `yes` if prompted. This may take a few minutes.
  
6. Active `mrai_p3` environment with:
```python
conda activate mrai_p3
```
7. Install `MRAi_P3` with:
```python
pip install MRAi_P3-1.0.0-py3-none-any.whl
```
* Installation may take a few minutes.
  
8. Install Jupyter with:
```python
conda install jupyter
```
* Type `yes` if prompted. This may take a few minutes.
  
9. Once completed, close the terminal.

---

### Installation on Apple Silicon (M1/M2/M3) — local, native

> The original wheel pins `tensorflow==2.13` and `numpy==1.24`,
> which **crash** on Apple Silicon with
> `The TensorFlow library was compiled to use AVX instructions… Abort`.
> The fix is to use the **current** TensorFlow (2.15+) which ships
> native macOS arm64 wheels.

Two supported routes are given below; **pick one**.

#### Option A — pyenv / venv (minimal, no conda)

Requirements:
- [pyenv](https://github.com/pyenv/pyenv) (Homebrew: `brew install pyenv`)
- Apple Silicon Mac (M1/M2/M3)

```bash
# 1. install a supported Python
pyenv install 3.12          # TF 2.21 supports 3.10–3.13
pyenv virtualenv 3.12 mrai_p3
mkdir -p ~/projects/tf_playground && cd $_
pyenv local mrai_p3          # writes .python-version

# 2. upgrade pip / tooling
python -m pip install --upgrade pip setuptools wheel

# 3. install TensorFlow for Apple Silicon (native cpu + gpu support)
pip install tensorflow       # >=2.13  ✓ arm64

# 4. install MRAi_P3 from the patched wheel for TensorFlow 2.15+ / Keras 3
pip install ~/git/mrai_p3/MRAi_P3-1.1.1-py3-none-any.whl

# 5. install Jupyter (optional, for notebooks)
pip install jupyter
```

#### Option B — conda / mambaforge (recommended if you prefer conda)

Requirements:
- [Miniforge](https://github.com/conda-forge/miniforge/releases) (for M-series Macs)

```bash
# create & activate a conda env with Python 3.12
conda create -n mrai_p3 python=3.12 -y
conda activate mrai_p3

# install TensorFlow (conda-forge keeps native arm64 wheels)
pip install tensorflow

# install MRAi_P3 from the patched wheel for TensorFlow 2.15+ / Keras 3
pip install /full/path-to/MRAi_P3-1.1.1-py3-none-any.whl

# Jupyter
conda install jupyter -y
```

#### Post-install sanity check (both options)
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import MRAi_P3 as P3; print('ok')"
```

---

## Post-installation Check
1. Open Terminal in the `MRAi_P3-main` folder.
2. Active `mrai_p3` environment with:
```python
conda activate mrai_p3
```
3. Start Python:
```python
python
```
4. In the Python prompt, run:
```python
import tensorflow as tf
```
5. If no error appears, the installation is correct.
6. If you see errors such as:
|`The TensorFlow library was compiled to use AVX instructions...`|
|Abort`
This means `MRAi_P3` is not compatible with your machine (e.g. Apple Silicon M-series Macs).
7. In case of incompatibility, we recommend running `MRAi_P3` on [NMRbox](https://nmrbox.org), where all dependencies are supported.

---

## Compatibility Notes (v1.1.1)

The `MRAi_P3-1.1.1` wheel includes patches for **TensorFlow 2.15+ / Keras 3**:

1. **SavedModel loading** — `tf.keras.models.load_model()` is replaced with `tf.keras.layers.TFSMLayer` wrapped in a `Sequential` model, since Keras 3 no longer supports the legacy SavedModel directory format.

2. **`model.predict()` output** — Keras 3 `TFSMLayer.predict()` returns a dict (e.g., `{'dense_14': array}`), so the output key is now extracted before assignment.

3. **NMRPipe `showhdr` fallback** — When NMRPipe's `showhdr` binary is not installed, `P5toPeaks.py` and `PeakstoTab.py` fall back to `nmrglue` to extract spectrum metadata (dimension count, nucleus labels, PPM ranges, sweep widths) from the Pipe dictionary.

---

## Additional Requirement
To access peak lists in NMRPipe format, the [NMRPipe](https://www.ibbr.umd.edu/nmrpipe) package must be installed separately. Without NMRPipe, peak list export in `.peaks` (CYANA) and `.tab` (Pipe) formats falls back to `nmrglue` for metadata extraction. The `.P5` and `.npy` peak list outputs are always available regardless of NMRPipe. We recommend running `MRAi_P3` on [NMRbox](https://nmrbox.org), where all dependencies are supported.

> **Note (v1.1.1):** The `3Dto2Dproj.sh` script (for 3D → 2D projections) requires `pipe2xyz` and `proj3D.tcl` from the NMRPipe suite. Without NMRPipe installed, only the P3 probability map and `.P5` peak list will be generated for 3D data.

---

## Usage
1. Open Terminal in `MRAi_P3-main` folder.
2. Active `mrai_p3` environment with:
```python
conda activate mrai_p3
```
3. Run Jupyter Notebook with:
```python
jupyter notebook
```
4. Open and run Jupyter notebook (`P3_Notebook.ipynb`), following the instructions provided in the file.

5. If necessary, restart the kernel and then shut down any open Jupyter instances before running a new notebook.

---

## Uninstallation
If it is needed to uninstall the entire environment, open a terminal in the `base` Conda environment and remove the `mrai_p3` environment with:
```python
conda remove --name mrai_p3 --all
```
