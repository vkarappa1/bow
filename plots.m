function [preprocessed]  = data_preprocess(data, k, 
preprocessed = []
parfor i=1:length(asl_caio)
  j = 1;
  x = [];
  k = 20;
  if sum(asl_caio(i,:)) > 0
    while j <= 450
      next = j + k - 1;
      x = [x (sum(asl_caio(i,j:next))/k)];
      j = j + k;
    end
    preprocessed = [preprocessed;x];
  end
end
end

