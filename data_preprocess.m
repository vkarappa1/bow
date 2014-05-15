function [preprocessed]  = data_preprocess(data, k)
preprocessed = [];
parfor i=1:length(data)
  j = 1;
  x = [];
  if sum(data(i,:)) > 0
    while j <= 450
      next = j + k - 1;
      x = [x (sum(data(i,j:next))/k)];
      j = j + k;
    end
    x = [x data(i,463)];
    preprocessed = [preprocessed;x];
  end
end
end

