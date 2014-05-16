close all;
clear all;
load caio.mat;

% preprocess and smoothen the data%
slp = data_preprocess(asl_caio, 50);
nslp = data_preprocess(non_asl_caio, 50);


opts = statset('Display','final', 'MaxIter', 1000, 'UseParallel','always');
k = 32;
slp_end = 80000;
nslp_end = 80000;
X = [slp(1:slp_end,1:end-1);nslp(1:nslp_end,1:end-1)];
[idx,ctrs,sumd] = kmeans(X,k,'Distance','sqEuclidean','Options',opts,'emptyaction','singleton', 'start', 'uniform', 'replicates', 1);


slp_bow = bow(slp(1:slp_end,end),idx(1:slp_end,:),k);
nslp_bow = bow(nslp(1:nslp_end,end),idx(slp_end + 1:end,:),k);

clusts_slp_test = vector_quantize(slp(slp_end + 1:end,1:end -1), ctrs, k);
clusts_nslp_test = vector_quantize(nslp(nslp_end + 1:end,1:end -1), ctrs, k);

slp_test_bow = bow(slp(slp_end+1:end,end),clusts_slp_test, k);
nslp_test_bow = bow(nslp(nslp_end+1:end,end),clusts_nslp_test, k);

 training = [slp_bow;nslp_bow];
 labels = [ones(size(slp_bow,1),1); ones(size(nslp_bow,1),1)*2];
 svmstruct = svmtrain(training,labels, 'Kernel_Function','linear');
  
assigna = svmclassify(svmstruct,slp_test_bow);
assignn = svmclassify(svmstruct,nslp_test_bow(1:63,:));
truep_test = length(find(assigna==1));
falsep_test = length(find(assignn==1));
falsen_test = length(find(assigna==2));
  
 
pre = truep_test/(truep_test + falsep_test)
rec = truep_test/(truep_test + falsen_test)
f = (2*pre*rec)/(pre + rec)