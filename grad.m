function best_w = grad(yx, C, eps, eta)
	w = sparse(size(yx,2),1);
	gd_0 = (grad_one(w, yx, C));
	eps_gd_0_norm = eps * norm(gd_0);
	%fprintf('gd_0 norm %f\n',eps_gd_0_norm);
	w = w - eta * gd_0;
	gd_norm = 1e10;
	while(gd_norm > eps_gd_0_norm)
		gd = grad_one(w, yx, C);
		w = w - eta * gd;
		gd_norm = norm(gd);
		%fprintf('gd norm %f\n',gd_norm);
	end
	best_w = w;
end
function gd = grad_one(w, yx, C)
	gd = w + C * transpose(yx) * (1./(1+exp(-(yx*w)))-1);
end
