# Fp-Classification :raising_hand:

## 0. Introduction

- In this basic project, I tried to classify 5 football players(classes) using simple form of AlexNet. 


## 1. Required Packages

- matplotlib==3.3.4
- numpy==1.20.0
- opencv-python==4.5.1.48
- scikit-learn==0.24.1
- torch==1.7.1
- torchvision==0.8.2
- tqdm==4.56.0


## 2. Tutorial :smiley:

## 2.1 Crawl data

- Using google images to get the data. You just need to make a file, `crawl_data.py` for example, then add into it:

```

from google_images_download import google_images_download  #importing the library 
response = google_images_download.googleimagesdownload()  #class instantiation 
arguments = {"keywords":"neymar","limit":100,"print_urls":True}  #creating list of arguments 
paths = response.download(arguments)  #passing the arguments to the function 


print(paths)

```
 
- Save & Run `python3 crawl_data.py` to get **100** images of **Neymar**. Do the same things with other players.

- Note that this tool is not always going well because of the uncertainty of Google Image(you'll see!). But anyway, check the data later :smiley:



### 2.2 Usage :raising_hand:

- Following these steps:
  - Create a folder named `data_after_splitting` and run the file `split_data.py` to split data into 2 common sets : train & val
  - Run `pip3 install -r requirements.txt` to get all required packages.
  - Run `preprocessing.py` to get encoded data
  - Run `train.py` 


### 3. Todo :sleepy:

