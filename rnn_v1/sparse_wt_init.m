

% Step3: Set architecture
arch_name = strcat(num2str(Nh),'N',num2str(dout),'L')
posN = strfind(arch_name,'N');
posL = strfind(arch_name,'L');
posS = strfind(arch_name,'S');
posR = strfind(arch_name,'R');
posM = strfind(arch_name,'M');
posfull = sort([posN,posL,posS,posR,posM]);
posfull = [0 posfull];

nl = [din];
for i = 1:length(posfull)-1
    nl = [nl str2num(arch_name(posfull(i)+1:posfull(i+1)-1))];
    f(i) = arch_name(posfull(i+1));
    
    switch f(i)
        case 'N'
            ff{i} = @nntanh;
            bf{i} = @dernntanh;
        case 'S'
            ff{i} = @nnsigm;
            bf{i} = @dernnsigm;
        case 'L'
            ff{i} = @nnlin;
            bf{i} = @dernnlin;
        case 'R'
            ff{i} = @nnrelu;
            bf{i} = @dernnrelu;
        case 'M'
            ff{i} = @nnsmax;
            bf{i} = @dernnsmax;
    end
    
end
arch_name = strcat(arch_name,'_rnn');
nl
f
ff
bf
nh = length(nl) - 1; % number of hidden layers
pflag = 0;

% sparse initialization 
W = 0.1*randn(Nh);
for i = 1:Nh;     W(i,randperm(Nh,Nh-20)) = 0; end;
W = W*(1.1/abs(eigs(W,1,'lm',opts)));

Wi = 0.1*randn(nl(2),nl(1));
U = 0.1*randn(dout,Nh);
bh = 0.1*randn(Nh,1);
bo = 0.1*randn(dout,1);

% move the parameters on to GPU
GWi = gpuArray(single(Wi));
GW = gpuArray(single(W));
GU = gpuArray(single(U));
Gbh = gpuArray(single(bh));
Gbo = gpuArray(single(bo));
GpdWi = gpuArray(single(zeros(size(Wi))));
GpdW = gpuArray(single(zeros(size(W))));
GpdU = gpuArray(single(zeros(size(U))));
Gpdbh = gpuArray(single(zeros(size(bh))));
Gpdbo = gpuArray(single(zeros(size(bo))));

GpmsgWit = gpuArray(single(zeros(size(Wi))));
GpmsgWt = gpuArray(single(zeros(size(W))));
GpmsgUt = gpuArray(single(zeros(size(U))));
Gpmsgbht = gpuArray(single(zeros(size(bh))));
Gpmsgbot = gpuArray(single(zeros(size(bo))));
GpmsxWit = gpuArray(single(zeros(size(Wi))));
GpmsxWt = gpuArray(single(zeros(size(W))));
GpmsxUt = gpuArray(single(zeros(size(U))));
Gpmsxbht = gpuArray(single(zeros(size(bh))));
Gpmsxbot = gpuArray(single(zeros(size(bo))));
