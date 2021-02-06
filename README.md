# Fp-Classification :raising_hand:

## 0. Introduction

- In this basic project, I try to classify 5 football players(classes) using AlexNet. 

- With ...

## 1. Required Packages

- Torch
- Cv2
- ...

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



### 2.2 Usage :sleepy:



## 3. Todo


## 4. Contributors

