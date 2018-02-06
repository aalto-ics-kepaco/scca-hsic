function obj = f(Kx,cKy)
%% to calculate objection function values for hsicCCA
N = size(Kx,1);
obj = trace(Kx*cKy)/(N-1)^2;
end