function [idx] = vector_quantize(data, ctrs, k)
  idx = [];
  parfor i=1:length(data)
    dists = [];
    for j=1:k
      diff = data(i,:) - ctrs(j,:);
      distance = sqrt(sum(diff.^2));
      dists = [dists; distance];
    end
    [v, m] = min(dists);
    idx = [idx; m];
  end
end