function [delbp] = nnls(ol_m,Out,derf)

delbp  = (ol_m - Out).*derf;

end