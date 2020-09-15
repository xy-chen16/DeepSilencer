library(gkmSVM)

# dataset to load
posfn= '/home/gaozijing/gzj/DeepSilencer/data/svm_6_23/train_pos.fa'
negfn= '/home/gaozijing/gzj/DeepSilencer/data/svm_6_23/train_neg.fa'
testfn= '/home/gaozijing/gzj/DeepSilencer/data/svm_6_23/test.fa'

# create output file folder
outpath = './saved_data'
dir.create(outpath)

# save path
kernelfn= './saved_data/test_kernel.txt'
svmfnprfx= './saved_data/test_svmtrain'
outfn =   './saved_data/output.txt'

# generate kernel
gkmsvm_kernel(posfn, negfn, kernelfn)

# train
gkmsvm_trainCV(kernelfn,posfn, negfn, svmfnprfx,nCV=5)

# classify and save the result
gkmsvm_classify(testfn, svmfnprfx, outfn)