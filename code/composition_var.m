% October 24, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Compute Compositional Variance                               %
%                                                                %
%================================================================%

function [norm_var, totvar] = composition_var(data)

% this function computes Aitchson's measures of variance

n = size(data,1);

norm_var = zeros(n,n);
for i = 1:n
    for j = 1:n
        norm_var(i,j) = var((1/(2^(.5)))*log(data(i,:)./data(j,:)));
    end
end
totvar = sum(sum(norm_var,1))/n;
end
