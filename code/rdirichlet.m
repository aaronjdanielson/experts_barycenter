% October 25, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Compute Compositional Variance                               %
%                                                                %
%================================================================%

function samp = rdirichlet(alpha,n)
 r = gamrnd(repmat(alpha,n,1),1,n,length(alpha)); 
 samp = r(:,1:end) ./ repmat(sum(r,2),1,length(alpha));
end