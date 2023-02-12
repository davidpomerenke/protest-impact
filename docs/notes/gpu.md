# runpod.io

```
apt-get update
apt-get install -y python3.10 python3.10-distutils python3.10-venv python3-pip
curl https://bootstrap.pypa.io/get-pip.py | python3.10
# python3.10 -m venv .venv
# .venv/bin/pip install poetry
# .venv/bin/poetry install
```

# Gradient

```
!git clone https://<token>@github.com/davidpomerenke/protest-impact-data.git .
!git pull
!pip install poetry
!poetry export -f requirements.txt --output requirements.txt --without-hashes
!poetry export -f requirements.txt --output requirements.txt --without-hashes
```

# Colab

Based on https://gist.github.com/DPaletti/da7729c35ba7f9274c3635849608b7bc.

```
from google.colab import drive
import os
import sys
drive.mount('/content/drive/')
os.chdir('/content/drive/My Drive/0-protest-impact')
!git clone https://<token>@github.com/davidpomerenke/protest-impact-data.git
!apt-get install python3.10 python3.10-distutils
!update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
!update-alternatives --set python3 /usr/bin/python3.10
!curl -sS https://bootstrap.pypa.io/get-pip.py | python
!pip install poetry
!poetry config virtualenvs.in-project true
project_root = "/content/drive/My Drive/0-protest-impact/protest-impact-data"
os.chdir(project_root)
!poetry install --no-ansi
sys.path.append("/root/.cache/pypoetry/virtualenvs/protest-impact-nhf8wJf3-py3.10/lib/python3.10/site-packages")
sys.path.append(project_root)
```
