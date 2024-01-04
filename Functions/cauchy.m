function C = cauchy(x, y)
% Cauchy distribution, method from L. Devroye's book pp.201.
if nargin == 1, y = x; end
	for i = 1:x
		for j = 1:y
			while 1
				U = rand(1)*2-1;
				V = rand(1)*2-1;
				if (U*U+V*V)<=1.0 break; end
			end
			if U~=0
				C(i,j) = V/U;
			else
				C(i,j) = 0;
			end
		end
	end

