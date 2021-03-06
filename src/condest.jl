using GenericSchur;

function fun2m_condest(f, A, B, C; wp = 128)

  A = mp(A, wp);
  B = mp(B, wp);
  C = mp(C, wp);

  FA = schur(A);
  FB = schur(B);

  C = FA.Z' * C * FB.Z;

  # size of the perturbation, make sure it's sufficiently far given the 
  # current precision level. 
  ep = 10.0^(-wp/2.0);

  TA = FA.T + diagm(randn(size(FA.T, 1))) * ep * norm(FA.T, Inf);
  TB = FB.T + diagm(randn(size(FB.T, 1))) * ep * norm(FB.T, Inf);
  Y = diag_fun(f, TA, TB, C);
  
  h = sqrt(ep);

  EA = mp(randn(size(TA, 1), size(TA, 1)), wp); EA = EA / norm(EA) * h * norm(TA);
  EB = mp(randn(size(TB, 1), size(TB, 1)), wp); EB = EB / norm(EB) * h * norm(TB);

  # We force the same sparsity pattern on the perturbations
  # EA = (TA .!= 0) .* EA;
  # EB = (TB .!= 0) .* EB;

  TA = TA + EA;
  TB = TB + EB;

  Y2 = diag_fun(f, TA, TB, C);

  c = convert(Float64, norm(Y2 - Y) / norm(Y) / h)

  return c;

end
