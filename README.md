# Analysis-of-Twin-SVM-on-44-Binary-datasets

Code belongs to the paper: "**Comprehensive Evaluation of Twin SVM based Classifiers on UCI Datasets**", (Accepted in **Applied Soft Computing, Elsevier**)

This repository provides code for binary datasets. For multi-class datasets, go to the following repository:
https://github.com/Chandan-IITI/Analysis-of-Twin-SVM-on-46-multi_class-datasets

Please follow the below mentioned steps to reproduce the results of the paper:

1. Download the datasets from "http://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz"
2. Each folder is for one classifier and each folder has three .m extension files. For the execution and get the result you just need 
   to run the all_eval.m. However, you need to modify only one line in the remaining two folders kfold_eval.m and seperate_eval.m before      execution as follows:
   
    Add the path of your downloaded datasets folder in the variable 'datapath' as follows:

   datapath = 'provide the path of your dataset';

   Note that just add the path of that folder not the path of any specific datasets.

3. All the mentioned results of the paper will be printed in an excel file 'all_results.xlsx'. 
