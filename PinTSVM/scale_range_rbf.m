%SCALE_RANGE Give a vector of scales
%
%     SIG = SCALE_RANGE(X,NR,NMAX)
%
% INPUT
%   X      Data matrix or dataset
%   NR     Number of scales (default = 20)
%   NMAX   Number of (random) points to consider (default = 500)
%
% OUTPUT
%   SIG    Vector of scale values
%
% DESCRIPTION
% Give a reasonable range of scales SIG for the dataset X. The largest
% scale is given first. If NR is given, the number of scales is NR.

% Copyright: D.M.J. Tax, D.M.J.Tax@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands
  
function sig = scale_range_rbf(x,nr,Nmax)

if nargin<3
	Nmax = 500;
end
if nargin<2
	nr = 20;
end

% Compute the distances
d = sqrt(sqeucldistm(x,x));

% Find the largest and the smallest distance:
dmax = max(d(:));
d(d<=1e-12) = dmax;
dmin = min(d(:));

% ... and compute the range:
%log10 = log(10);
%sig = logspace(log(dmax)/log10,log(dmin)/log10,nr);
sig = linspace(dmax,dmin,nr);

return

