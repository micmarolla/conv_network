function z = softmax(y)
    f = exp(y-max(y));
    z = f / sum(f);
end