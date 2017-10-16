function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for example_index = 1:size(X,1)
  example = X(example_index, :);
  distance_list = zeros(K,1);
  for centroid_index = 1:K
    
    centroid = centroids(centroid_index, :);
    #distance = sum(sumsq(example - centroid));
    distance = (example - centroid) * (example - centroid)'
    distance_list(centroid_index, 1) = distance;
  endfor
  [closest_cen, closest_cen_index] = min(distance_list)
  idx(example_index) = closest_cen_index
endfor





% =============================================================

end

