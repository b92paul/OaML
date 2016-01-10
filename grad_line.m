function best_w = grad_line(yX, C, eps, eta)
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
		
		%s = -gd;
		CG_iter = 0;
		% line search
		alpha = 1.0;
		sTs = dot(gd,gd);
		eta_gdTs = -eta * sTs;
		
		wTs = -dot(w,gd);
		
		ayXs = -(yX*gd);

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
		w = w - alpha * gd;
		wTw = dot(w,w);
		% for next step
		fprintf('iter  %d f %0.3e |g| %0.3e CG   %d step_size %0.3e\n', t_iter, f_k, gd_norm, CG_iter, alpha);
		%fprintf('iter %d, gd_norm = %f, alpha = %f, wnorm = %f\n', t_iter, f_k, gd_norm, alpha, sqrt(wTw));
		yXw = yX*w;
		f_k = obj_func_f(wTw, yXw, C);
		sigm_nyXw = 1./(1+exp(-yXw));
		gd = w + C * transpose((transpose(sigm_nyXw-1))*yX);
		gd_norm = norm(gd);
		t_iter = t_iter+1;
	end
	best_w = w;
end
function v = obj_func_f(wTw, yXw, C)
	v = 0.5*wTw + C*sum(log(1+exp(-yXw)));
end
function v = obj_func_fast(wTw, wTs, sTs, yXwPayXs, alpha, C)
	v = 0.5*(wTw + 2*alpha*wTs + (alpha^2)*sTs) + C*sum(log(1+exp(-yXwPayXs)));
end
