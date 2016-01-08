function best_w = newton_line(yX, C, X, eps, eta, eps_k)
	w = zeros(size(yX,2),1);
	wTw = 0;
	yXw = yX*w;
	sigm_nyXw = 1./(1+exp(-yXw)); %sigmf(-yXw,[1,0]);
	gd = w + C * transpose((transpose(sigm_nyXw-1))*yX);
	gd_norm = norm(gd);
	gd_0_norm = gd_norm;
	t_iter = 1;
	f_k = obj_func_f(wTw, yXw, C);
	while(gd_norm > eps * gd_0_norm)
		diagD = sigm_nyXw .* (1-sigm_nyXw);
		[s, CG_iter] = my_CG(gd, eps_k, X, diagD, C);
		% line search
		alpha = 1.0;
		eta_gdTs = eta * dot(gd,s);
		
		sTs = dot(s,s);
		wTs = dot(w,s);
		
		ayXs = yX*s;

		while(1)
			f_was = obj_func_fast(wTw, wTs, sTs, yXw + ayXs, alpha, C);
			f_wPgd = f_k + alpha*eta_gdTs;
			if(f_was < f_wPgd)
				break;
			end
			alpha = alpha * 0.5;
			ayXs = ayXs * 0.5;
		end
		
		% update w
		w = w + alpha * s;
		wTw = dot(w,w);
		% for next step
		f_k = obj_func_f(wTw, yXw, C);
		fprintf('iter  %d f %f |g| %f CG   %d step_size %f\n', t_iter, f_k, gd_norm, CG_iter, alpha);
		%fprintf('iter %d, gd_norm = %f, alpha = %f, wnorm = %f\n', t_iter, f_k, gd_norm, alpha, sqrt(wTw));
		yXw = yX*w;
		sigm_nyXw = 1./(1+exp(-yXw));
		gd = w + C * transpose((transpose(sigm_nyXw-1))*yX);
		gd_norm = norm(gd);
		t_iter = t_iter+1;
	end
	best_w = w;
end
function [s, CG_iter] = my_CG(gd, eps_k, X, diagD, C) % gd = d*1, X = l*d, diagD = l*1
	s = zeros(size(gd, 1),1);
	r = -gd;
	d = -gd;
	gd_norm = norm(gd);
	r_norm = gd_norm;
	CG_iter = 0;
	while(r_norm > eps_k*gd_norm )
		Xd = X*d; % (l*d)*(d*1) = (l*1), bad part ...
		DXd = diagD .* Xd; % l*1
		DXd_T = transpose(DXd);
		a = (r_norm^2) / ( C*dot(Xd,DXd) + dot(d,d));
		s = s + a * d;
		r = r - a * (d + C*transpose(DXd_T*X));
		r_norm_new  = norm(r);
		beta = (r_norm_new / r_norm)^2;
		d = r + beta*d;
		r_norm = r_norm_new;
		CG_iter = CG_iter+1;
	end
end
function v = obj_func_f(wTw, yXw, C)
	v = 0.5*wTw + C*sum(log(1+exp(-yXw)));
end
function v = obj_func_fast(wTw, wTs, sTs, yXwPayXs, alpha, C)
	v = 0.5*(wTw + 2*alpha*wTs + (alpha^2)*sTs) + C*sum(log(1+exp(-yXwPayXs)));
end
