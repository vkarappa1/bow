close all;
clear all;
load caio.mat;

clusters  = [8 16 32]; %number of clusters to be used%
resolutions = [10 20 30 40 50]; %resolution of PMP%
slCount = 60; %count videos for training from each set%
iterations = 100; %number of cluster and random video iterations%
opts = statset('Display','final', 'MaxIter', 1000, 'UseParallel','always');

precisions_reso  = [];
recalls_reso = [];
f1s_reso = [];

for r=1:length(resolutions) %for every resolution%
  
  precisions  = [];
  recalls = [];
  f1s = [];
  
  % preprocess and smoothen the data%
  resolution = resolutions(r);
  slp = data_preprocess(asl_caio, resolution); 
  nslp = data_preprocess(non_asl_caio, resolution);

  for l=1:length(clusters)
    
    k = clusters(l)
    precs = [];
    recs =  [];
    fs = [];
  
    parfor iter=1:iterations
      
      [slp_train, slp_test, slp_train_vids] = getRandom(slp,slCount);
      [nslp_train, nslp_test, slp_test_vids] = getRandom(nslp,slCount);
      
      slp_end = length(slp_train); % used to index of the clusts_train returned by the kmeans
      nslp_end = length(nslp_train);
      X = [slp_train(:,1:end - 1);nslp_train(:,1:end - 1)]; % till resolution because the last column is the video id%
      [clusts_train,ctrs,sumd] = kmeans(X,k,'Distance','sqEuclidean','Options',opts,'emptyaction','singleton', 'start', 'uniform', 'replicates', 1);
      
      %training bag of words%
      slp_train_bow = bow(slp_train(:,end),clusts_train(1:slp_end,:),k);
      nslp_train_bow = bow(nslp_train(:,end),clusts_train(slp_end + 1:end,:),k);
      
      %vector quanitzzation of test frames%
      clusts_slp_test = vector_quantize(slp_test(:,1:end -1), ctrs, k);
      clusts_nslp_test = vector_quantize(nslp_test(:,1:end -1), ctrs, k);
      
      %testing bag of words%
      slp_test_bow = bow(slp_test(:,end),clusts_slp_test, k);
      nslp_test_bow = bow(nslp_test(:,end),clusts_nslp_test, k);
      
      training = [slp_train_bow;nslp_train_bow];
      labels = [ones(size(slp_train_bow,1),1); ones(size(nslp_train_bow,1),1)*2];
      svmstruct = svmtrain(training,labels, 'Kernel_Function','linear');
      
      assigna = svmclassify(svmstruct,slp_test_bow);
      assignn = svmclassify(svmstruct,nslp_test_bow);
      truep_test = length(find(assigna==1));
      falsep_test = length(find(assignn==1));
      falsen_test = length(find(assigna==2));
      
      pre = truep_test/(truep_test + falsep_test)
      rec = truep_test/(truep_test + falsen_test)
      f = (2*pre*rec)/(pre + rec)
      
      precs = [precs; pre];
      recs = [recs; rec];
      fs = [fs; f];
    end
    
    precisions = [precisions mean(precs)];
    recalls = [recalls mean(recs)];
    f1s = [f1s mean(fs)];
  
  end
  
  precisions_reso = [precisions_reso; precisions];
  recalls_reso = [recalls_reso; recalls];
  f1s_reso = [f1s_reso; f1s];

  save('results.mat', 'precisions_reso', 'recalls_reso', 'f1s_reso');
  
end