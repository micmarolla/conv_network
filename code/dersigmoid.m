function y = dersigmoid(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
end