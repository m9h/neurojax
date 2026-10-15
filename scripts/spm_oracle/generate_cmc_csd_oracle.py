"""Run SPM's CSD forward model for the test-6 architectures entirely in MATLAB and save a MAT file."""
import sys
from spm_runtime import endpoint
out = sys.argv[1]
code = r'''
res = struct();
for arch = {'single','four'}
  for perturb = [0 1]
    if strcmp(arch{1},'single')
      A = {0,0,0}; B = {0}; C = 1; X = sparse(1,0);
    else
      A = {zeros(4),zeros(4),zeros(4)}; A{1}(2,1)=1; A{1}(4,3)=1; A{2}(1,2)=1; A{2}(3,4)=1;
      A{3}(3,1)=1; A{3}(1,3)=1; A{3}(4,2)=1; A{3}(2,4)=1;
      B = {double((A{1}+A{2}+A{3})>0)}; C = [1;0;1;0]; X = sparse([0;1]);
    end
    [P,pC] = spm_dcm_neural_priors(A,B,C,'CMC');
    Ns = size(A{1},1);
    dipfit = struct('model','CMC','type','LFP','location',0,'Ns',Ns,'Nc',Ns);
    [P,pC] = spm_L_priors(dipfit,P,pC);
    P = spm_ssr_priors(P,pC);
    if perturb
      P.G = P.G + 0.2; P.T = P.T - 0.1; P.D = P.D + 0.3; P.a = P.a + 0.5;
    end
    [x,f] = spm_dcm_x_neural(P,'CMC');
    M = struct(); M.model='CMC'; M.f=f; M.g='spm_gx_erp'; M.x=x; M.n=length(spm_vec(x)); M.m=Ns;
    M.u=sparse(Ns,1); M.l=Ns; M.Hz=(2:64)'; M.dt=1/128; M.dipfit=dipfit;
    U = struct(); U.X = X;
    [y,w,s] = spm_csd_mtf(P,M,U);
    key = sprintf('%s_%d',arch{1},perturb);
    res.(key).y = y; res.(key).P = P; res.(key).Hz = M.Hz; res.(key).X = full(X);
    res.(key).s = s;
  end
end
save(OUTFILE,'-struct','res','-v7');
'''
endpoint("eval", code.replace("OUTFILE", "'" + out + "'"), nargout=0)
print("saved", out)
