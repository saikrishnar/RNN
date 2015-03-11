function [y] = nntanh(ac)

a_tanh = 1.7159;
b_tanh = 2/3;
y = a_tanh*tanh(b_tanh*ac);

end