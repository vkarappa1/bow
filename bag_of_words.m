
 vidid_asl = aslpolarforall(:,463);
  uvidid_asl = unique(vidid_asl);
  train_asl_bag_of_words = [];
  for i=1:numel(uvidid_asl)
    ids = find(vidid_asl == uvidid_asl(i));
    xvalues1 = 1:k;
    nelements = hist(idx_train_asl(ids),xvalues1);
    train_asl_bag_of_words =  [train_asl_bag_of_words; nelements];
  end
  bw_asl = mean(train_asl_bag_of_words);
  