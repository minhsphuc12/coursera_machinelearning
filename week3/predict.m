function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
m
# 16
num_labels
# 4

size(X)
# 16*2
a1 = [ones(m, 1) X];
# 16*3

size(Theta1)
# 4*3

#a2 = (sigmoid(X * Theta1') >= 0.5);
#a2 = [ones(num_labels, 1) a2];

z2 = a1 * Theta1';
# 16*4
a2 = [ones(size(z2),1) sigmoid(z2)];
# 16*5
size(Theta2);
# 4*5
z3 = a2 * Theta2';
# 16*4
a3 = sigmoid(z3);
#16*4

[val p] = max(a3, [], 2);

% =========================================================================


end
