
datadir = strcat(expdir,'data/');
load(strcat(datadir,'train.mat'));

% Step1: make training data
train_batchdata = gpuArray(single(X));
tmvnf = 1;
% Input file preprocessing
if tmvnf
    m = mean(data);
    v = std(data);
    train_batchdata = bsxfun(@minus,train_batchdata,m);
    train_batchdata = bsxfun(@rdivide,train_batchdata,v+1e-5);
    save(strcat(datadir,'mvn.mat'),'m','v');
end

train_batchtargets = gpuArray(single(Y));
train_clv = cumsum([1 sv]);
train_numbats = length(train_clv) - 1;
[Nin,din] = size(train_batchdata)
[Nout,dout] = size(train_batchtargets)
clear X Y sv

% Step2: make validation data
load(strcat(datadir,'val.mat'));
val_batchdata = gpuArray(single(X));
if tmvnf
    val_batchdata = bsxfun(@minus,val_batchdata,m);
    val_batchdata = bsxfun(@rdivide,val_batchdata,v+1e-5);
end

val_batchtargets = gpuArray(single(Y));
val_clv = cumsum([1 sv]);
val_numbats = length(val_clv) - 1;
clear X Y sv

% Step3: make test data
load(strcat(datadir,'test.mat'));
test_batchdata = gpuArray(single(X));
if tmvnf
    test_batchdata = bsxfun(@minus,test_batchdata,m);
    test_batchdata = bsxfun(@rdivide,test_batchdata,v+1e-5);
end

test_batchtargets = gpuArray(single(Y));
test_clv = cumsum([1 sv]);
test_numbats = length(test_clv) - 1;
clear X Y sv
