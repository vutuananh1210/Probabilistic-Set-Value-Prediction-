
## 1 Prepare data: 
From the root of the project, run the script "python -m scripts.get_data --task MCC" to get the probabilistic prediction from random forest for *all* datas in the folder MCC. Or for some specific datasets "python -m scripts.get_data --task MCC --datasets glass yeast". Change the flag to MDC or MLC accordingly for the other types. 

## 2.
File analysis/current_work.py contains the main code. File analysis/test.ipynb is a notebook for further analysis if needed.  
