function [hsic_nystrom] = f_nystrom(phix, phiy)

hsic_nystrom = norm(1/size(phix,1) * phix' * phiy,'fro')^2;

end

