C = 0.1;
e = 0.1;
eta = 0.01;

%[y, x] = libsvmread('data/heart_scale');
[y, x] = libsvmread('data/kddb_test_2');
y = y*2-1;
fprintf('read file done.\n');
y_diag = spdiags(y ,0,size(y,1),size(y,1));
yx = y_diag * x;
fprintf('start training ...\n');
best_w = grad_line(yx, C, e, eta);
%best_w = grad(yx, C, e, eta);
ans = sum((y-(2*(myeval(best_w,x)>=0.5)-1))==0)
