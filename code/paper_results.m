% October 25, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Test Script                       %
%                                                                %
%================================================================%

% Not production software - use to execute code

n = 7;
K = 6;
alpha= [.10,.20,.35,.15,.10,.10,
   .05,.40,.05,.40,.05,.05,
   .01,.01,.01,.01,.01,.95,
   .55,.10,.10,.21,.02,.02,
   .75,.05,.05,.05,.05,.05,
   .25,.50,.075,.05,.075,.05,
   .01,.01,.01,.11,.01,.85];

 w = ones(n,1)/n; % uniform weights
 D = ones(K) - eye(K); % Hamming distance
 
 mygam = dwbary(alpha,w,D);
 bcenter = plot_dwbary(mygam)
 sum(alpha)/7
 
 composition_var(alpha)
 
 
 for t = 1:6
     sum(mygam(:,t,:))
 end
 
 
 % for the paper: sample the opinion of M experts R times from
 % a dirichlet distribution with uniform weights
 % how would this compare to the normal-logisitic? 
 
 %a = ones(K,1)';
 %n = 10;
 %r = gamrnd(repmat(a,n,1),1,n,length(a)); % or perhaps call randg
 %r = r(:,1:end) ./ repmat(sum(r,2),1,length(a));
 
 R = 100;
 K = 6;
 % parameters for the uniform dirichlet
 a1 = ones(K,1)';
 n1 = 10;
 % parameters for the extreme draws
 a2 = [1,1,1,1,1.5,10].^4;
 n2 = 1;
 ameans = zeros(R,K);
 bmeans = zeros(R,K);
 maxdat = zeros(R,2);
 mindat = zeros(R,2);
 w = ones((n1+n2),1)/(n1+n2); % uniform weights
 D = ones(K) - eye(K); % Hamming distance
 for r = 1:R
     dat1 = rdirichlet(a1,n1);
     dat2 = rdirichlet(a2,n2);
     dat = [dat1;dat2];
     maxes = max(dat,[],1);
     mines = min(dat,[],1);
     ameans(r,:) = sum(dat)/(n1+n2);
     [g,b] = dwbary(dat,w,D);
     bmeans(r,:) = b;
     % record distance to the boundary in a single dimension
     %maxdat(r,:) = [abs(b-maxes), abs(sum(dat)/n - maxes)];
     %mindat(r,:) = [abs(b-mines), abs(sum(dat)/n - mines)]; 
 end
 
  max(ameans,[],1);
  max(bmeans,[],1);
  min(ameans,[],1);
  min(bmeans,[],1);
  var(ameans,[],1);
  var(bmeans,[],1);
  var(ameans,[],2);
  var(bmeans,[],2);
  
  [a1,a2] = composition_var(ameans);
  [b1,b2] = composition_var(bmeans);
  
  % this seems to be the direction ...
  
  sum(ameans(:,6) - bmeans(:,6)>0);
  
  % the arithmetic mean of the outlying expert is usually larger than
  % the others, 88 out of 100 samples. 
  
  hold on
  scatter(ameans(:,1),ameans(:,2),"red")
  scatter(bmeans(:,1),bmeans(:,2),"blue")
  
  hold on
  scatter(ameans(:,1),ameans(:,6),"red")
  scatter(bmeans(:,1),bmeans(:,6),"blue")
  
  
  % show asymptotic distribution for barycenter
  % show it is less efficient than the arithmetic mean
  % show it is more robust when distributional assumptions fail
  % that is, the barycenter may be more useful in the wild
 
 