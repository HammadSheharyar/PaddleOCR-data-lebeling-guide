## CPU Version Guide for labelling Cropped BIB images with PaddleOCR Label, using PPOCRv5det and korean_ppocrv5_rec models

# ist install all dependinces and models, run both det and extraction models, 
# it will run draw bboxes and extract all text present in the image, 
# we need to manually remove the extra bbox and text extracted , only leave bib number and korean/english name
# save the labels for det model training and save the rec result
# the bbox will text doc, and rec will be 2 cropped images saved along with text doc for results

# create separate conda enviremnent , python >= 3.11.15 
# activate
# install dependices , and paddle lable tool

conda activate yolo_gemma


python --version

 <!-- bcz we have gpu-->
pip install paddlepaddle-gpu


gpu version can not work here bcz this is old gpu, compute 5.0, so we downloa the cpu

https://www.paddlepaddle.org.cn/packages/stable/cu126/paddlepaddle-gpu/?utm_source=chatgpt.com
paddlepaddle-3.3.1-cp311-cp311-win_amd64.whl

pip install paddlepaddle-3.3.1-cp311-cp311-win_amd64.whl

python -c "import paddle; print(paddle.__version__)"
3.3.1



# Install PaddleOCR
pip install paddleocr



python -c "import paddle; paddle.utils.run_check()"
INFO: Could not find files for the given pattern(s).
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
Running verify PaddlePaddle program ...
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\pir\math_op_patch.py:241: UserWarning: Tensor do not have 'place' interface for pir graph mode, try not to use it. None will be returned.
  warnings.warn(
I0817 16:29:30.281266   608 pir_interpreter.cc:1529] New Executor is Running ...
I0817 16:29:30.285408   608 pir_interpreter.cc:1552] pir interpreter is running by multi-thread mode ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.


python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
INFO: Could not find files for the given pattern(s).
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
3.3.1
False
cpu




(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> pip install "paddlepaddle-3.3.1-cp311-cp311-win_amd64.whl"
Processing .\paddlepaddle-3.3.1-cp311-cp311-win_amd64.whl
Collecting httpx (from paddlepaddle==3.3.1)
  Using cached httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
Requirement already satisfied: numpy>=1.21 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlepaddle==3.3.1) (2.4.4)
Collecting protobuf>=3.20.2 (from paddlepaddle==3.3.1)
  Using cached protobuf-7.35.1-cp310-abi3-win_amd64.whl.metadata (595 bytes)
Requirement already satisfied: Pillow in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlepaddle==3.3.1) (12.2.0)
Collecting opt-einsum==3.3.0 (from paddlepaddle==3.3.1)
  Using cached opt_einsum-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: networkx in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlepaddle==3.3.1) (3.6.1)
Requirement already satisfied: typing-extensions in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlepaddle==3.3.1) (4.15.0)
Collecting safetensors>=0.6.0 (from paddlepaddle==3.3.1)
  Downloading safetensors-0.8.0-cp310-abi3-win_amd64.whl.metadata (4.2 kB)
Collecting anyio (from httpx->paddlepaddle==3.3.1)
  Using cached anyio-4.14.2-py3-none-any.whl.metadata (4.6 kB)
Requirement already satisfied: certifi in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from httpx->paddlepaddle==3.3.1) (2026.6.17)
Collecting httpcore==1.* (from httpx->paddlepaddle==3.3.1)
  Using cached httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
Requirement already satisfied: idna in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from httpx->paddlepaddle==3.3.1) (3.18)
Collecting h11>=0.16 (from httpcore==1.*->httpx->paddlepaddle==3.3.1)
  Using cached h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
Using cached opt_einsum-3.3.0-py3-none-any.whl (65 kB)
Using cached protobuf-7.35.1-cp310-abi3-win_amd64.whl (439 kB)
Downloading safetensors-0.8.0-cp310-abi3-win_amd64.whl (355 kB)
Using cached httpx-0.28.1-py3-none-any.whl (73 kB)
Using cached httpcore-1.0.9-py3-none-any.whl (78 kB)
Using cached h11-0.16.0-py3-none-any.whl (37 kB)
Using cached anyio-4.14.2-py3-none-any.whl (125 kB)
Installing collected packages: safetensors, protobuf, opt-einsum, h11, anyio, httpcore, httpx, paddlepaddle
Successfully installed anyio-4.14.2 h11-0.16.0 httpcore-1.0.9 httpx-0.28.1 opt-einsum-3.3.0 paddlepaddle-3.3.1 protobuf-7.35.1 safetensors-0.8.0
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> python -c "import paddle; print(paddle.__version__)"
INFO: Could not find files for the given pattern(s).
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
3.3.1
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> pip install paddleocr
Collecting paddleocr
  Downloading paddleocr-3.7.0-py3-none-any.whl.metadata (28 kB)
Collecting paddlex<3.8.0,>=3.7.0 (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading paddlex-3.7.2-py3-none-any.whl.metadata (80 kB)
Requirement already satisfied: PyYAML>=6 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddleocr) (6.0.3)
Requirement already satisfied: requests in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddleocr) (2.34.2)
Collecting aiohttp>=3.8.0 (from paddleocr)
  Downloading aiohttp-3.14.3-cp311-cp311-win_amd64.whl.metadata (8.5 kB)
Requirement already satisfied: typing-extensions>=4.12 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddleocr) (4.15.0)
Collecting aistudio-sdk>=0.3.5 (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading aistudio_sdk-0.3.9-py3-none-any.whl.metadata (1.2 kB)
Collecting chardet (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading chardet-7.6.0-cp311-cp311-win_amd64.whl.metadata (9.6 kB)
Collecting colorlog (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading colorlog-6.12.0-py3-none-any.whl.metadata (11 kB)
Requirement already satisfied: filelock in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (3.29.0)
Collecting huggingface-hub (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading huggingface_hub-1.27.0-py3-none-any.whl.metadata (16 kB)
Collecting modelscope>=1.28.0 (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading modelscope-1.39.1-py3-none-any.whl.metadata (43 kB)
Collecting numpy<2.4,>=1.24 (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading numpy-2.3.5-cp311-cp311-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: packaging in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (26.0)
Collecting pandas>=1.3 (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading pandas-3.0.5-cp311-cp311-win_amd64.whl.metadata (19 kB)
Requirement already satisfied: pillow in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (12.2.0)
Collecting prettytable (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading prettytable-3.18.0-py3-none-any.whl.metadata (37 kB)
Collecting py-cpuinfo (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
Collecting pydantic>=2 (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached pydantic-2.13.4-py3-none-any.whl.metadata (109 kB)
Collecting PyYAML>=6 (from paddleocr)
  Downloading PyYAML-6.0.2-cp311-cp311-win_amd64.whl.metadata (2.1 kB)
Collecting ruamel.yaml (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached ruamel_yaml-0.19.1-py3-none-any.whl.metadata (16 kB)
Collecting ujson (from paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading ujson-5.13.0-cp311-cp311-win_amd64.whl.metadata (10 kB)
Collecting imagesize (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading imagesize-2.0.0-py2.py3-none-any.whl.metadata (1.5 kB)
Collecting opencv-contrib-python==4.10.0.84 (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl.metadata (20 kB)
Collecting pyclipper (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading pyclipper-1.4.0-cp311-cp311-win_amd64.whl.metadata (8.8 kB)
Collecting pypdfium2>=4 (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading pypdfium2-5.13.0-py3-none-win_amd64.whl.metadata (67 kB)
Collecting python-bidi (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading python_bidi-0.6.11-cp311-cp311-win_amd64.whl.metadata (5.4 kB)
Collecting shapely (from paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading shapely-2.1.2-cp311-cp311-win_amd64.whl.metadata (7.1 kB)
Collecting aiohappyeyeballs>=2.5.0 (from aiohttp>=3.8.0->paddleocr)
  Downloading aiohappyeyeballs-2.7.1-py3-none-any.whl.metadata (5.9 kB)
Collecting aiosignal>=1.4.0 (from aiohttp>=3.8.0->paddleocr)
  Using cached aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
Collecting attrs>=17.3.0 (from aiohttp>=3.8.0->paddleocr)
  Using cached attrs-26.1.0-py3-none-any.whl.metadata (8.8 kB)
Collecting frozenlist>=1.1.1 (from aiohttp>=3.8.0->paddleocr)
  Downloading frozenlist-1.8.0-cp311-cp311-win_amd64.whl.metadata (21 kB)
Collecting multidict<7.0,>=4.5 (from aiohttp>=3.8.0->paddleocr)
  Downloading multidict-6.7.1-cp311-cp311-win_amd64.whl.metadata (5.5 kB)
Collecting propcache>=0.2.0 (from aiohttp>=3.8.0->paddleocr)
  Downloading propcache-0.5.2-cp311-cp311-win_amd64.whl.metadata (17 kB)
Collecting yarl<2.0,>=1.17.0 (from aiohttp>=3.8.0->paddleocr)
  Downloading yarl-1.24.5-cp311-cp311-win_amd64.whl.metadata (107 kB)
Requirement already satisfied: idna>=2.0 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from yarl<2.0,>=1.17.0->aiohttp>=3.8.0->paddleocr) (3.18)
Requirement already satisfied: psutil in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (7.2.2)
Collecting tqdm (from aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading tqdm-4.70.0-py3-none-any.whl.metadata (57 kB)
Collecting bce-python-sdk (from aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading bce_python_sdk-0.9.76-py3-none-any.whl.metadata (558 bytes)
Collecting click (from aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached click-8.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting modelscope-hub>=0.2.0 (from modelscope>=1.28.0->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading modelscope_hub-0.2.0-py3-none-any.whl.metadata (32 kB)
Requirement already satisfied: setuptools in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from modelscope>=1.28.0->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (82.0.1)
Requirement already satisfied: urllib3>=1.26 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from modelscope>=1.28.0->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (2.7.0)
Requirement already satisfied: python-dateutil>=2.8.2 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from pandas>=1.3->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (2.9.0.post0)
Collecting tzdata (from pandas>=1.3->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached tzdata-2026.3-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting annotated-types>=0.6.0 (from pydantic>=2->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading annotated_types-0.8.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.46.4 (from pydantic>=2->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading pydantic_core-2.46.4-cp311-cp311-win_amd64.whl.metadata (6.7 kB)
Collecting typing-inspection>=0.4.2 (from pydantic>=2->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading typing_inspection-0.4.4-py3-none-any.whl.metadata (2.6 kB)
Requirement already satisfied: six>=1.5 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from python-dateutil>=2.8.2->pandas>=1.3->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (1.17.0)
Requirement already satisfied: charset_normalizer<4,>=2 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from requests->paddleocr) (3.4.9)
Requirement already satisfied: certifi>=2023.5.7 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from requests->paddleocr) (2026.6.17)
Collecting colorama (from tqdm->aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting pycryptodome>=3.8.0 (from bce-python-sdk->aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached pycryptodome-3.23.0-cp37-abi3-win_amd64.whl.metadata (3.5 kB)
Collecting future>=0.6.0 (from bce-python-sdk->aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Using cached future-1.0.0-py3-none-any.whl.metadata (4.0 kB)
Collecting crc32c>=2.2.post0 (from bce-python-sdk->aistudio-sdk>=0.3.5->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading crc32c-2.8-cp311-cp311-win_amd64.whl.metadata (8.0 kB)
Requirement already satisfied: fsspec>=2023.5.0 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (2026.4.0)
Collecting hf-xet<2.0.0,>=1.5.2 (from huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading hf_xet-1.6.0-cp38-abi3-win_amd64.whl.metadata (4.9 kB)
Requirement already satisfied: httpx<1,>=0.23.0 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (0.28.1)
Requirement already satisfied: anyio in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (4.14.2)
Requirement already satisfied: httpcore==1.* in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from httpx<1,>=0.23.0->huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (1.0.9)
Requirement already satisfied: h11>=0.16 in C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages (from httpcore==1.*->httpx<1,>=0.23.0->huggingface-hub->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr) (0.16.0)
Collecting wcwidth>=0.3.5 (from prettytable->paddlex<3.8.0,>=3.7.0->paddlex[ocr-core]<3.8.0,>=3.7.0->paddleocr)
  Downloading wcwidth-0.8.2-py3-none-any.whl.metadata (43 kB)
Downloading paddleocr-3.7.0-py3-none-any.whl (146 kB)
Downloading paddlex-3.7.2-py3-none-any.whl (2.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 4.9 MB/s  0:00:00
Downloading PyYAML-6.0.2-cp311-cp311-win_amd64.whl (161 kB)
Downloading numpy-2.3.5-cp311-cp311-win_amd64.whl (13.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.1/13.1 MB 3.3 MB/s  0:00:03
Using cached opencv_contrib_python-4.10.0.84-cp37-abi3-win_amd64.whl (45.5 MB)
Downloading aiohttp-3.14.3-cp311-cp311-win_amd64.whl (481 kB)
Downloading multidict-6.7.1-cp311-cp311-win_amd64.whl (45 kB)
Downloading yarl-1.24.5-cp311-cp311-win_amd64.whl (97 kB)
Downloading aiohappyeyeballs-2.7.1-py3-none-any.whl (15 kB)
Using cached aiosignal-1.4.0-py3-none-any.whl (7.5 kB)
Downloading aistudio_sdk-0.3.9-py3-none-any.whl (67 kB)
Using cached attrs-26.1.0-py3-none-any.whl (67 kB)
Downloading frozenlist-1.8.0-cp311-cp311-win_amd64.whl (44 kB)
Downloading modelscope-1.39.1-py3-none-any.whl (6.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.0/6.0 MB 6.0 MB/s  0:00:01
Downloading modelscope_hub-0.2.0-py3-none-any.whl (156 kB)
Downloading pandas-3.0.5-cp311-cp311-win_amd64.whl (10.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.0/10.0 MB 2.7 MB/s  0:00:03
Downloading propcache-0.5.2-cp311-cp311-win_amd64.whl (42 kB)
Using cached pydantic-2.13.4-py3-none-any.whl (472 kB)
Downloading pydantic_core-2.46.4-cp311-cp311-win_amd64.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 5.0 MB/s  0:00:00
Downloading annotated_types-0.8.0-py3-none-any.whl (13 kB)
Downloading pypdfium2-5.13.0-py3-none-win_amd64.whl (3.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.9/3.9 MB 1.7 MB/s  0:00:03
Downloading tqdm-4.70.0-py3-none-any.whl (80 kB)
Downloading typing_inspection-0.4.4-py3-none-any.whl (14 kB)
Downloading bce_python_sdk-0.9.76-py3-none-any.whl (435 kB)
Downloading crc32c-2.8-cp311-cp311-win_amd64.whl (66 kB)
Using cached future-1.0.0-py3-none-any.whl (491 kB)
Using cached pycryptodome-3.23.0-cp37-abi3-win_amd64.whl (1.8 MB)
Downloading chardet-7.6.0-cp311-cp311-win_amd64.whl (1.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 2.9 MB/s  0:00:00
Using cached click-8.4.2-py3-none-any.whl (119 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Downloading colorlog-6.12.0-py3-none-any.whl (12 kB)
Downloading huggingface_hub-1.27.0-py3-none-any.whl (784 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 784.9/784.9 kB 2.6 MB/s  0:00:00
Downloading hf_xet-1.6.0-cp38-abi3-win_amd64.whl (4.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.0/4.0 MB 3.2 MB/s  0:00:01
Downloading imagesize-2.0.0-py2.py3-none-any.whl (9.4 kB)
Downloading prettytable-3.18.0-py3-none-any.whl (37 kB)
Downloading wcwidth-0.8.2-py3-none-any.whl (323 kB)
Using cached py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
Downloading pyclipper-1.4.0-cp311-cp311-win_amd64.whl (104 kB)
Downloading python_bidi-0.6.11-cp311-cp311-win_amd64.whl (163 kB)
Using cached ruamel_yaml-0.19.1-py3-none-any.whl (118 kB)
Downloading shapely-2.1.2-cp311-cp311-win_amd64.whl (1.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 3.0 MB/s  0:00:00
Using cached tzdata-2026.3-py2.py3-none-any.whl (348 kB)
Downloading ujson-5.13.0-cp311-cp311-win_amd64.whl (40 kB)
Installing collected packages: py-cpuinfo, wcwidth, ujson, tzdata, typing-inspection, ruamel.yaml, PyYAML, python-bidi, pypdfium2, pydantic-core, pycryptodome, pyclipper, propcache, numpy, multidict, imagesize, hf-xet, future, frozenlist, crc32c, colorama, chardet, attrs, annotated-types, aiohappyeyeballs, yarl, tqdm, shapely, pydantic, prettytable, pandas, opencv-contrib-python, colorlog, click, bce-python-sdk, aiosignal, modelscope-hub, huggingface-hub, aistudio-sdk, aiohttp, modelscope, paddlex, paddleocr
  Attempting uninstall: PyYAML
    Found existing installation: PyYAML 6.0.3
    Uninstalling PyYAML-6.0.3:
      Successfully uninstalled PyYAML-6.0.3
  Attempting uninstall: numpy
    Found existing installation: numpy 2.4.4
    Uninstalling numpy-2.4.4:
      Successfully uninstalled numpy-2.4.4
Successfully installed PyYAML-6.0.2 aiohappyeyeballs-2.7.1 aiohttp-3.14.3 aiosignal-1.4.0 aistudio-sdk-0.3.9 annotated-types-0.8.0 attrs-26.1.0 bce-python-sdk-0.9.76 chardet-7.6.0 click-8.4.2 colorama-0.4.6 colorlog-6.12.0 crc32c-2.8 frozenlist-1.8.0 future-1.0.0 hf-xet-1.6.0 huggingface-hub-1.27.0 imagesize-2.0.0 modelscope-1.39.1 modelscope-hub-0.2.0 multidict-6.7.1 numpy-2.3.5 opencv-contrib-python-4.10.0.84 paddleocr-3.7.0 paddlex-3.7.2 pandas-3.0.5 prettytable-3.18.0 propcache-0.5.2 py-cpuinfo-9.0.0 pyclipper-1.4.0 pycryptodome-3.23.0 pydantic-2.13.4 pydantic-core-2.46.4 pypdfium2-5.13.0 python-bidi-0.6.11 ruamel.yaml-0.19.1 shapely-2.1.2 tqdm-4.70.0 typing-inspection-0.4.4 tzdata-2026.3 ujson-5.13.0 wcwidth-0.8.2 yarl-1.24.5
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> python -c "import paddle; paddle.utils.run_check()"
INFO: Could not find files for the given pattern(s).
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
Running verify PaddlePaddle program ...
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\pir\math_op_patch.py:241: UserWarning: Tensor do not have 'place' interface for pir graph mode, try not to use it. None will be returned.
  warnings.warn(
I0817 16:29:30.281266   608 pir_interpreter.cc:1529] New Executor is Running ...
I0817 16:29:30.285408   608 pir_interpreter.cc:1552] pir interpreter is running by multi-thread mode ...
PaddlePaddle works well on 1 CPU.
PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
INFO: Could not find files for the given pattern(s).
C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddle\utils\cpp_extension\extension_utils.py:712: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
3.3.1
False
cpu
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations> nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Fri_Jun_14_16:44:19_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.20
Build cuda_12.6.r12.6/compiler.34431801_0
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations>


















# Install PPOCRLabel (the GUI tool)
pip install PPOCRLabel

# Verify
PPOCRLabel --help


python -c "from PPOCRLabel.PPOCRLabel import main; main()" --lang en --det_model_name PP-OCRv5_mobile_det --rec_model_name korean_PP-OCRv5_mobile_rec

OSError: [WinError 127] The specified procedure could not be found. Error loading "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\torch\lib\shm.dll" or one of its dependencies.
(yolo_gemma) PS F:\paddleocr-finetuned\data-annotations\dataset-2026>

  import modelscope
  File "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\modelscope\__init__.py", line 5, in <module>
    from modelscope.utils.import_utils import (LazyImportModule,
  File "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\modelscope\utils\import_utils.py", line 23, in <module>
    from modelscope.utils.ast_utils import (INDEX_KEY, MODULE_KEY, REQUIREMENT_KEY,
  File "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\modelscope\utils\ast_utils.py", line 23, in <module>
    from modelscope.utils.file_utils import get_modelscope_cache_dir
  File "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\modelscope\utils\file_utils.py", line 13, in <module>
    logger = get_logger()


<!--  -->
we edit models files

# before
import modelscope

# after
try:
    import modelscope
except ImportError:
    modelscope = None


 notepad "E:\anaconda3\envs\yolo_cpu\Lib\site-packages\paddlex\inference\utils\official_models.py"

 if this not opend

 python -c "import paddlex, os; print(os.path.join(os.path.dirname(paddlex.__file__), 'inference', 'utils', 'official_models.py'))"

we got the path

C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddlex\inference\utils\official_models.py


open in vscode

code "C:\Users\Pc\miniconda3\envs\yolo_gemma\Lib\site-packages\paddlex\inference\utils\official_models.py"



















































