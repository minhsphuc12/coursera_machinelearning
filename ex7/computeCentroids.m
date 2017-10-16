function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

for centroid_index = 1:K
  clusters = [];
  for example_cluster_index = 1:size(idx, 1)
    if (idx(example_cluster_index) == centroid_index)
      clusters = [clusters ; X(example_cluster_index, :)];
    endif
  endfor
  centroids(centroid_index, :) = mean(clusters, 1);
endfor 
  
    
    
# Much shorter solution from internet
#for i=1 :K
#  centroids(i, :) = mean(X([find(idx == i)], :));
#end




% =============================================================


end

