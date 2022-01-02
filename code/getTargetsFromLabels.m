function T = getTargetsFromLabels(labels)
    len = length(labels);
    ll = unique(labels)';
    numL = length(ll);
    T = zeros(numL, len);
    for n = 1:len
        T(labels(n)==ll,n) = 1;
    end
end