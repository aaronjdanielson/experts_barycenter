% October 24, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Plot Barycenter Probability Vector                        %
%                                                                %
%================================================================%

function [bcenter] = plot_dwbary(gam)
n = size(gam,1);
figure(1), clf;
for i = 1:n
    subplot(1,n,i), imagesc(squeeze(gam(i,:,:)));
end
bcenter = sum(gam,3);
bcenter = bcenter(1,:);
end