function best_w = grad_line(yx, C, eps, eta)
	t_iter = 0;
	%w = sparse(size(yx,2),1);
	w = zeros(size(yx,2),1);
	gd_0 = (grad_one(transpose(yx*w), w, yx, C));
	eps_gd_0_norm = eps * norm(gd_0);
	%fprintf('gd_0 norm %f\n',eps_gd_0_norm);
	gd_norm = 1e10;
	%size(yx)
	while(gd_norm > eps_gd_0_norm)
		t_iter = t_iter + 1;
		alpha = 1;
		yx_w = (yx*w);
		yx_w_T = transpose(yx_w);
		gd = grad_one(yx_w_T, w, yx, C);
		a_yx_gd = yx*gd;
		
		gd_norm = norm(gd);
		w_norm = norm(w);
		w_gd = dot(w, gd);
		
		eta_gd_norm2 = eta * (gd_norm^2);
		
		fw_k = obj_func_f(w_norm, yx_w, C); % f(w_k)
		iter = 0;
		while( obj_func_fast(w_norm, gd_norm, w_gd, yx_w, a_yx_gd, alpha, C) > fw_k - alpha * eta_gd_norm2)
			alpha = alpha * 0.5;
			a_yx_gd = a_yx_gd * 0.5;
			iter = iter+1;
		end
		if(mod(t_iter,10) == 0)
			fprintf('iter:%d, alpha iter = %d: alpha = %f, gd_norm = %f\n', t_iter, iter, alpha, gd_norm);
		end
		w = w - alpha * gd;
	end
	gd_norm = norm(gd);
	fprintf('total iter = %d, gd_norm = %f\n', t_iter, gd_norm);
	best_w = w;
end
function gd = grad_one(yx_w_T, w, yx, C)
	gd = w + C * transpose((1./(1+exp(-(yx_w_T)))-1)*yx);
end
function v = obj_func_f(w_norm, yx_w, C)
	v = 0.5*w_norm^2 + C*sum(log(1+exp(-yx_w)));
end
function v = obj_func_fast(w_norm, gd_norm, w_gd, yx_w, a_yx_gd, alpha, C)
	 v = 0.5*(w_norm^2 - 2*alpha*w_gd + (alpha*gd_norm)^2) + C*sum(log(1+exp(-yx_w+a_yx_gd)));
end
