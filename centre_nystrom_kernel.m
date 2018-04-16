function [phic] = centre_nystrom_kernel(phi)

phic = (eye(size(phi,1)) - 1 / size(phi,1) * ones(size(phi,1))) * phi;


end

