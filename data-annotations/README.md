## Guide for labelling Cropped BIB images with PaddleOCR Label, using PPOCRv5det and korean_ppocrv5_rec models

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


























































