function y = projL2(x,r)
y = x;
if norm(x) > 0.0001
    y = r * x / norm(x);
end
end