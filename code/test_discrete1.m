% October 24, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Compute Barycenter Probability Vector                        %
%                                                                %
%================================================================%


function  = dwbary(data, weights, distance)

% DWVARY Compute Discrete Wasserstein Barycenter for a Set of Probability
% Vectors
%
%         Use this function to:
%           1. Compute the Discrete Barycenter
%           2. Create plots


n = 6;
K = 6;
alpha= [.10,.20,.35,.15,.10,.10,
   .05,.40,.05,.40,.05,.05,
   .01,.01,.01,.01,.01,.95,
   .55,.10,.10,.21,.02,.02,
   .75,.05,.05,.05,.05,.05,
   .25,.50,.075,.05,.075,.05];
  %.15,.15,.15,.15,.15,.25]
  
  w = ones(n,1)/n; % uniform weights
  D = ones(K) - eye(K); % Hamming distance
  cost = zeros(n,K,K);
  for i = 1:n
      cost(i,:,:) = w(i)*D;
  end
  
  %%
  cvx_begin
    variable b(1,K)
    variable gam(n,K,K)
    
    minimize sum(gam(:).*cost(:))
    
    squeeze(sum(gam,2)) == alpha
    sum(gam,3) == repmat(b,n,1)
    
    gam >= 0
  cvx_end

end

  %%
  figure(1), clf
  for i = 1:n
      subplot(1,n,i), imagesc(squeeze(gam(i,:,:)))
  end
           