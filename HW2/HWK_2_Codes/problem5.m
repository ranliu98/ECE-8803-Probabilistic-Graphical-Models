function diprob5
import brml.*
load('diseaseNet_compat_BRML-ObjectOriented.mat');

pot=str2cell(setpotclass(pot,'array'));

[jtpot jtsep infostruct]=jtree(pot);
jtpot=absorption(jtpot,jtsep,infostruct);

for symp=21:60
    jtpotnum = whichpot(jtpot,symp,1);
    margpot=sumpot(jtpot{jtpotnum},symp,0);
    jtmarg(symp-20)=margpot.table(1);
end

for symp=1:40
    fprintf(1,'s[%d]=1 || %g',symp,jtmarg(symp));
    fprintf('\n');
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n');
fprintf('\n');
fprintf('\n');


effff = dag(pot);
for symp=21:60
    margpot2 = sumpot(multpots([pot(symp) pot(parents(effff,symp))]),symp,0);
    jtmarg2(symp-20)=margpot2.table(1);
end;

for symp=1:40
    fprintf(1,'s[%d]=1 || %g',symp,jtmarg2(symp));
    fprintf('\n');
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[jtpot jtsep]=jtassignpot(setpot(pot, [21:30],[1 1 1 1 1 2 2 2 2 2]),infostruct);
jtpot=absorption(jtpot,jtsep,infostruct);
for dise=1:20
    jtpotnum = whichpot(jtpot,dise,1)
    margpot=condpot(sumpot(jtpot(jtpotnum),dise,0));
    jtwithevidence(dise)=margpot.table(1);
    fprintf(1,'d[%d]=1 under condition =%g\n',dise,jtwithevidence(dise));
end
