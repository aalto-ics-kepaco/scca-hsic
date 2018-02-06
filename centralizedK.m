function K = centralizedK(K)

K= K+ mean(K(:)) -(repmat(mean(K,1),[size(K,1),1])+repmat(mean(K,2),[1,size(K,2)]) );
end
