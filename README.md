# Virtual Environment Setup Guide

This guide explains how to set up a virtual environment, install necessary libraries, and keep the environment up to date using `requirements.txt`.

## **1. Initial Setup**

### 1.1 Create the Virtual Environment
In VS code, press ctrl + ` to open a terminal. Make sure you are in your project directory.

Run:
```bash
python -m venv env696
```
### 1.2 Activate the Virtual Environment
```bash
env696\Scripts\activate
```
### 1.3 Install Required Libraries
Use the requirements.txt file to install libraries:
```bash
pip install -r requirements.txt
```
## **2. Working with the Virtual Environment in VScode Jupyter**
### 2.1 Every Time You Start Working
Activate the virtual environment in VScode jupyter:

Press this button:

![image](./readme_pic/123.png)

Change to `env696`:

![image](./readme_pic/456.png)

Verify activation by running this in jupyter:
```bash
!where python
```
The first path should be within `env696`.
### 2.2 When There Are Changes to `requirements.txt`
Run:
```bash
pip install -r requirements.txt
```

### 2.3 Every Time You Install or Update a Library
Export to the `requirements.txt`:
```bash
pip freeze > requirements.txt
```
If your code require downloaded model, add the download command near that block.
![image](./readme_pic/789.png)

## 3. Notes
`env696` folder is created under the project folder, but it will not upload to github, as I include `.gitignore` to tell git to ignore this folder.

If you encounter any issues or want to start fresh:
1. Manually delete the `env696` folder.
2. Recreate the environment by following step 1 again.

# Project Timeline

Ideally, the tasks with same stage can be done parallel.

| Stage | Task | Assign to |
| --- | --- | --- |
| 1 | Data cleaning, merging | Derick |
| 2 | Text processing | Derick |
| 2 | Normalization, One hot encoding, format consistent | Hinson |
| 3 | Topic modeling | Derick |
| 3 | Clustering | Hinson |
| 4 | Interpret topic modelling result | Both |
| 5 | Clustering (continue with topic modelling result) | Hinson |
| 6 | Predict popularity, score (traditional model) | Hinson |
| 6 | Predict popularity, score (deep learning) | Derick |
| 7 | Compare model and extract feature importance | Both |
| 7 | Forecast trend | if time sufficient |

### How to get update from github

click the fetch button

![image](https://github.com/user-attachments/assets/cbcb280c-ed41-42eb-9140-cdcf0af06c0c)

A new commit will show up in the source control graph if it has new update.

Then, click sync change.

![image](https://github.com/user-attachments/assets/f4226f8e-9c7d-488f-bed2-0f4d64134d97)


### How to view diffs

- view diffs for current commit (before you push):
  
  In VScode, save the notebook and click the arrow surround button
  
  ![image](https://github.com/user-attachments/assets/172f5f35-2f4b-44a0-bbe8-3e56236cd402)
  
  ![image](https://github.com/user-attachments/assets/c400e61e-e712-4628-a5c6-e61cc7b53af1)

- view diffs for all previous commit:
  
  VScode don't have good support for viewing jupyter diffs 
  ![image](https://github.com/user-attachments/assets/23d5329d-9dfb-4dc2-a988-06f1b86c6295)

  So, use third party website, provide user friendly view:
  https://app.reviewnb.com/derickkan3356/696/
  ![image](https://github.com/user-attachments/assets/ebd860cd-6195-470d-a1db-4ad93aed8aa2)


