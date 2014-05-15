function [train, test, train_vids] = getRandom(data, count)
ids = unique(data(:,end));
train_vids = ids(randperm(length(ids), count));
length(train_vids)
train = [];
indices = [];
test = data;
for id=1:train_vids
  index = find(data(:,end) == id);
  train = [train; data(index,:)];
  indices = [indices; index];
end
test(indices,:) = [];
end