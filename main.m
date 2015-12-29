C = 0.1;
e = 0.1;
eta = 0.001;

[y, x] = libsvmread('data/heart_scale');
y_diag = spdiags(y ,0,size(y,1),size(y,1));
yx = y_diag * x;
best_w = grad(yx, C, e, eta)
ans = sum((y-(2*(myeval(best_w,x)>=0.5)-1))==0)