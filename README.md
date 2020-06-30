# DeepSilencer
#### A deep convolutional neural network for the accurate prediction of silencers
<div align=center>
<img src = "inst/Figure1.png" width = 80% height = 80%>
</div>  

## Installation  

```  
Requiements:  
1. Python 3.5 or later version  
2. Packages:  
    numpy (>=1.15.1)  
    keras (2.3.1)  
    tensorflow(-gpu) (1.15.2)  
    hickle (>=3.4)
  
Package installation:
  
$ pip install -U numpy  
$ pip install keras == 2.3.1 
$ pip install tensorflow-gpu==1.15.2 #pip install tensorflow==1.15.2  
$ pip install -U hickle  
$ git clone https://github.com/xy-chen16/DeepSilencer.git   
$ cd DeepSilencer    
```

## Data Preprocessing

### Load the genome files:
```  
$ cd data 
$ mkdir -p genome/mm10 && cd genome/mm10
$ nohup wget http://hgdownload.cse.ucsc.edu/goldenPath/mm10/bigZips/chromFa.tar.gz
$ tar zvfx chromFa.tar.gz
$ cd ..
$ mkdir hg19 && cd hg19
$ nohup wget http://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz
$ tar zvfx chromFa.tar.gz 
$ cd ../../..
```
### Unzip the open region files and result files:
```  
$ tar -xjvf data/open_region.tar.bz2 -C data
$ tar -xjvf result/result.tar.bz2 -C result
```
## Tutorial  
### self-projection 
```   
$ python code/run_self_projection.py
```
<div align=center>
<img src = "inst/Figure2.png" width = 60% height = 60%>
</div>  
