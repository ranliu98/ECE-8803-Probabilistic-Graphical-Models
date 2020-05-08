function p7
import brml.*
load('p.mat');
p=setpotclass(p,'array');
%% 

% approximating q structure:
[x y z]=assign(1:3);
qxy = array([x y],normp(rand([3 3])));
qz = array(z,normp(rand([3,1])));
%% 

% mean field:
for loop=1:50
    qxy = condpot(exppot(sumpot(multpots([logpot(p) qz]),z)));
    qz = condpot(exppot(sumpot(multpots([logpot(p) qxy]),[x y])));
    qxyz = multpots([qxy qz]);
    kl(loop) = KLdiv(qxyz,p);
    plot(kl,'-o');
    title('KL divergence'); drawnow
    fprintf(1,'kl in %g is %g\n',loop,kl(loop))
end