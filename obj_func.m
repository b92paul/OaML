function v = obj_func(w, yx, C)
	v = 0.5*norm(w)^2 + C*sum(log(1+exp(-yx*w)));
end
