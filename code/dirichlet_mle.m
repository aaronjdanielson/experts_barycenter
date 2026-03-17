% October 24, 2018
%================================================================%
%        Discrete Barycenter for Probability Vectors             %    
%================================================================%
%                                                                %
%   Dirichlet MLE computation                                    %
%                                                                %
%================================================================%

n = 6;
K = 6;
alpha= [.10,.20,.35,.15,.10,.10,
   .05,.40,.05,.40,.05,.05,
   .01,.01,.01,.01,.01,.95,
   .55,.10,.10,.21,.02,.02,
   .75,.05,.05,.05,.05,.05,
   .25,.50,.075,.05,.075,.05];

alpha= [.10,.20,.35,.15,.10,.10,
   .10,.35,.05,.40,.05,.05,
   .10,.01,.01,.01,.01,.86,
   .10,.55,.10,.21,.02,.02,
   .10,.7,.05,.05,.05,.05,
   .10,.50,.075,.20,.075,.05];

alpha= [.11,.19,.35,.15,.10,.10,
   .10,.35,.05,.40,.05,.05,
   .10,.01,.01,.01,.01,.86,
   .10,.55,.10,.21,.02,.02,
   .10,.7,.05,.05,.05,.05,
   .10,.50,.075,.20,.075,.05];

beta = sum(log(alpha))/size(alpha,2);

custnloglf = @(lambda,data,cens,freq) - size(data,1)*(log(gamma(sum(lambda))) - sum(log(gamma(lambda))) + sum((lambda-1)*data));

%Estimate the parameters of the defined distribution.
options = optimset('MaxIter',200000,"MaxFunEvals",200000);
phat = mle(beta,'nloglf',custnloglf,"start",[2 2 2 2 2 2],'optimOptions', options);
phat/sum(phat)
phat = mle(beta,'nloglf',custnloglf,"start",[.5 .2 1 5 2 20],'optimOptions', options);
phat/sum(phat)
%phat = mle(beta,'nloglf',custnloglf,'start',0.05);