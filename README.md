# Basler

https://www.baslerweb.com/en/downloads/software/ 에서 pylon설치 파일 다운.
사용 버전 : pylon 6.3.0 Camera Software Suite Linux x86 (64 Bit) - Debian Installer Package

## Installation Using Debian Packages

The Debian package will always install the pylon 6 Camera Software Suite in the
/opt/pylon directory. On many Debian-based Linux distributions, you can install
the Debian package by double-clicking the file. Alternatively, follow these
these steps:

1. Change to the directory that contains the pylon Debian package.

2. Install the Debian package:
   sudo apt-get install ./pylon\_\*.deb

3. python package  
   pip install pypylon

# Sam2PE

## Sam2

보다 자세한 문서는 [sam/README.md](./sam/README.md) 확인

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd sam
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

(note that these are the improved checkpoints denoted as SAM 2.1; see [Model Description](#model-description) for details.)

## Getting started

## Installation

python 3.10 이상의 환경에서 사용.

```
cd sam2pe
# poetry 사용하는 경우
poetry install
# poetry 미사용
pip install -e .
```

## Usage

scripts/sam2stream.py  
--source SOURCE, -s SOURCE : path to source directory  
--template_dir TEMPLATE_DIR, -t TEMPLATE_DIR : directory to save the output masks (as PNG files)  
--generate, -g : Set generating template mode. Save template masks to TEMPLATE_DIR.
--roi : path to roi.txt
--test TEST : path to test directory

template 생성 및 test 코드.  
roi 미입력 시 roi 지정
SOURCE의 이미지들 중 일부 선택 후 template 생성
TEST directory 안의 image들로 segmentation & 각도 측정. TEST_results 에 결과 저장.

```bash

# template 생성
python .\scripts\sam2stream.py --source .\dataset\btn_1129\btn00\ --template_dir .\dataset\templates\btn00 -g
python .\scripts\sam2stream.py --template_dir .\dataset\templates\under_btn03 -g
# 저장된 이미지로 test
python .\scripts\sam2stream.py  --template_dir .\dataset\templates\btn00 --test  .\dataset\btn_1129\btn00\
```


scripts/rmc_pe.py  
-- template_dir TEMPLATE_DIR, -t TEMPLATE_DIR : path to the template directory  
-- out_dir OUT_DIR, -o OUT_DIR : path to save the input, cropped, output images

```bash
python scripts/rmc_pe.py -t .\dataset\templates\btn10 -o results/btn10
python .\scripts\rmc_pe.py -t .\dataset\templates\btn02 -o results/btn02 --source camera
```

#### template 생성

SOURCE에 있는 이미지 중 지정해서 template mask 생성.

1. 's' : template 지정할 frame 선택
2. center 영역 지정
3. end point 영역 지정

### Template 구성

```
template*dir/
   L *.jpg : color image
   L 001/ : object*id -> object center
   L *.png : mask image
   L 002/ : object_id -> end point area
   L *.png : mask image
   L roi.txt
```
