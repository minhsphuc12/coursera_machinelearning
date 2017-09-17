function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

##### Forward feedback (or forward propagation)
# we already has m is row size of X, so we add column of 1 (bias unit) to X
# that will be activation of input layer
a1 = [ones(m, 1) X];
# forward to hidden layer
z2 = a1 * Theta1';
# a1 = m * input_size + 1 ; Theta1' = input_size+1 * hidden_size
# calculate activation of hidden layer, adding column of 1 (bias unit) to 
# that activation vector
a2 = [ones(size(z2,1),1) sigmoid(z2)];
# forward to output layer
z3 = a2 * Theta2';
# calculate activation of output layer
a3 = sigmoid(z3);
# calculate cost function without regularization term
# this is based on activation of output layer and y vector alone
# to effectively compare output of m prediction to m observation output, 
# y must be turned to square matrix, with each row is one observation,
# and column has one represent class that observation belongs to.
Y = eye(num_labels)(y,:)
#J = (1/m) * sum(sum((-Y .* log(a3)) - ((1-Y) .* log(1 - a3)),2));
#cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
#J = (1 / m) * sum(cost);
J = (-sum(sum(Y.*log(a3))) - sum(sum((1-Y).*(log(1-a3)))))/m;

## calculate regularization term
# this is based on weight alone
# this does not include weight of bias unit
Theta1_nobias = Theta1(:,2:end);
Theta2_nobias = Theta2(:,2:end);
reg_term = lambda/(2*m) * (sum(sumsq(Theta1_nobias)) + sum(sumsq(Theta2_nobias)));
J = J + reg_term
#regularization_term = (lambda/(2*m))*((sum(sum(Theta1(:,2:end).^2))) + sum(sum(Theta2(:,2:end).^2)));
#J = J + regularization_term;

# initiate error matrix of output layer and hidden
#delta3 = zeros(size(Y))
#delta2 = zeros(size(sigmoid(z2)))
# (a) calculate error of output layer
delta3 = a3 - Y;
# (b) calculate error of hidden layer
#delta2 = delta3 * Theta2 .* sigmoidGradient(a2);
delta2 = delta3 * Theta2 .* [ones(size(z2,1),1) sigmoidGradient(z2)];
# why use bias term and sigmoidGradient of z2, instead of sgGrad of a2
# Theta2' is hidden_size * num_labels ; delta3 is m * num_labels
# delta2 is m * hidden_size

## base on (a), calculate gradient of hidden layer
# size of that gradient is theta2, which is num_labels * hidden_size +1
# size of delta3 is m * num_labels; size of a2 is m * hidden_size+1
D_2 =  (delta3)' * a2;
# base on (b), calculate gradient of input layer, but not use error of bias term
D_1 = delta2(:,2:end)' * a1;
# D_1 is theta1, which is hidden_size * input_size +1 
# delta 2 is m * hidden_size, X is m* input_size
# performance gradient descent on weight, without regularization term 
Theta1_grad = Theta1_grad + 1/m * D_1;
Theta2_grad = Theta2_grad + 1/m * D_2;
# for neuron not bias unit, add regularization term (without second power)
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda/m * Theta1_nobias;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda/m * Theta2_nobias;













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
