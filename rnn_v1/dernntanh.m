function [df] = dernntanh(ac)
a_tanh = 1.7159;
b_tanh = 2/3;
bby2a = (b_tanh/(2*a_tanh));
df = bby2a*((a_tanh - ac).*(a_tanh + ac));
end
