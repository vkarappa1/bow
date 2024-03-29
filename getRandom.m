function [train, test, train_vids] = getRandom(data, count)
ids = unique(data(:,end));
train_vids = ids(randperm(length(ids), count));
train = [];
indices = [];
test = data;
for id=1:count
  index = find(data(:,end) == train_vids(id));
  train = [train; data(index,:)];
  indices = [indices; index];
end
test(indices,:) = [];
end