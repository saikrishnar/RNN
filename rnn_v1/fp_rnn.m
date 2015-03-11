function [hcm,ym] = fp_rnn(X,Wi,W,U,bh,bo,ff,Nh,sl)

hcm = gpuArray(zeros(sl,Nh));
hp = gpuArray(zeros(Nh,1));

X = X';
is = bsxfun(@plus,Wi*X, bh);

for k = 1:sl    
    % forward prop
    ac = W*hp + is(:,k);
    hc = arrayfun(ff{1},ac);
    hp = hc;
    % store params for T time steps
    hcm(k,:) = hc';    
end

ym = bsxfun(@plus,U*hcm',bo)';
    

end
