function HMMprob2
import brml.*
load('prob2inter.mat');
load('noisystring.mat');

geID = [82,83,84]
stID = [1,6,11,15,18,23,29,35,38,43,54,65]

[alpha, loglik]=HMMforward(noisystring,trans,prior,emiss);
[sigma,logpvhstar]=HMMviterbi(noisystring,trans,prior,emiss);

pnum=zeros(1,length(sigma));
gnum=zeros(1,length(sigma));
for i=1:length(sigma)
    k=find(sigma(i)==stID);
    if ~isempty(k); pnum(i)=k; end
    l=find(sigma(i)==geID);
    if ~isempty(l); gnum(i)=l; end
    if isempty(k)&isempty(l)
        pnum(i)=pnum(i-1);
        gnum(i)=gnum(i-1);
    end
end

pcount=zeros(12,12);
fname=false; sname=false;

for i=1:length(v)
    if pnum(i)>0 
        if gnum(i-1)==1;
        fname=true; fnamep=pnum(i);
        end
        if gnum(i-1)==2;
        sname=true; snamep=pnum(i);
        end
        
        if fname && sname
            pcount(fnamep,snamep)=pcount(fnamep,snamep)+1;
            fname=false; sname=false;
        end
    end
end
pcount