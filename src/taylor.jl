
function findk(f, A, B, C, delta, maxk = 170)
	# Compute the total degree of the truncated bivariate Taylor expansion
	# for approximating f{A, B}(C) with accuracy delta, for triangular A and B
	#
	#--------------------------INPUT-------------------------------------------
	#
	# f			handle function for evaluating f and its derivatives: f(x, y, i, j)
	# A,B		mxm and nxn triangular matrix coefficients (likely the Schur forms)
	# C 		mxn right hand side
	# delta     upper bound for the magnitude of the residual
	# maxk		(optional) maximum value for k, default = 170
	#
	#--------------------------------------------------------------------------
	#

	(m, n) = size(C);
	lA = diag(A); lB = diag(B);
	l = sum(lA) / m; mu = sum(lB) / n;
	nrmC = norm(C);
	nrmA = norm(diag(A) .- l, Inf); # opnorm(A - l * one(A), 1);
	nrmB = norm(diag(B) .- mu, Inf); # opnorm(B - mu * one(B), 1);

	k = 0;

  ff = norm([ f(x,y,0,0) for x = lA, y = lB ], Inf)

	while k < maxk
		k = k + 1;
		y = findk_res(f, l, mu, k, nrmA, nrmB);
		if y < delta * ff
			mx = -Inf;
			for x = lA  # TODO it might be done more efficiently if f(x, y, i, j) allows vectorization
				for y = lB
					tmp = findk_res(f, x, y, k, nrmA, nrmB);
					if tmp > mx
						mx = tmp;
					end
				end
			end
			if mx < delta * ff
				break
			end
		end
	end

	return k
end

#------------------Auxiliary function for findk --------------------------------
function findk_res(f, x, y, k, nrmA, nrmB)
	yy = 0.0;

	fj = 1.0;
	fk = convert(Float64, factorial(big(k+1)));

	for j = 0 : k + 1
		# yy = yy + abs(f(x, y, k + 1 - j, j)) / factorial(k + 1 - j) * nrmA^(k + 1 - j) / factorial(j) * nrmB^j;
		yy = yy + abs(f(x, y, k + 1 - j, j)) / fk * nrmA^(k + 1 - j) / fj * nrmB^j;

		# Update factorials for the next round
		fj = (j+1) * fj;
		fk = fk / (k+1+j);
	end

	return yy;
end


function taylor_eval(f, A, B, C, delta)
# Returns [X, digits]
# Evaluate the truncated bivariate taylor expansion of accuracy delta of f{A, B}(C)
# for triangular coefficients A and B
#
#--------------------------INPUT-------------------------------------------
#
# f			handle function for evaluating f and its derivatives: f(x, y, i, j)
# A,B		mxm and nxn triangular matrix coefficients (likely the Schur forms)
# C 		mxn right hand side
# delta     upper bound for the magnitude of the residual
#
#--------------------------------------------------------------------------
#

	(m, n) = size(C);
	l = sum(diag(A)) / m; mu = sum(diag(B)) / n;
	k = findk(f, A, B, C, delta);
  digits = k;
	A = A - l * one(A);
	B = B - mu * one(B);

  coef = zeros(ComplexF64, k, k)
  for i = 1 : k
    for j = 1 : k
      if i + j <= k + 1
        coef[i, j] = f(l, mu, i - 1, j - 1)
      end
    end
  end

  # XX at the generic step is A^i / i! * C
  XX = C
  X = zero(C)

  for i = 1 : k

    YY = XX
    for j = 1 : k + 1 - i
      X = X + coef[i,j] * YY
      YY = YY * B / j
    end

    XX = A * XX / i
  end

	return X, digits;
end
