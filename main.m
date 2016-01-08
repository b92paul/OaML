C = 0.1;
eps = 0.01;
eta = 0.01;
eps_k = 0.1;

tic;
%[y, x] = libsvmread('data/heart_scale');
%[y, x] = libsvmread('data/rcv1_test');
[y, x] = libsvmread('data/kddb');
y = y*2-1;
toc;
fprintf('read file done.\n');
y_diag = spdiags(y ,0,size(y,1),size(y,1));
yx = y_diag * x;
fprintf('start training ...\n');
%tic;best_w = grad_line(yx, C, eps, eta);toc;
tic;best_w = newton_line(yx, C, x, eps, eta, eps_k);toc;
%tic;best_w = grad(yx, C, e, eta);toc;
ans = sum((y-(2*(myeval(best_w,x)>=0.5)-1))==0)
v = obj_func(best_w, yx, C)
