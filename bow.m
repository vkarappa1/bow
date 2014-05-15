function [bag_of_words] = bow(ids, clusts, k)
 u_ids = unique(ids);
 bag_of_words = [];
 for i=1:numel(u_ids)
  found_ids = find(ids == u_ids(i));
  xvalues1 = 1:k;
  nelements = hist(clusts(found_ids),xvalues1);
  bag_of_words =  [bag_of_words; nelements];
 end
end