function A=p3ChowLiu(X)
import brml.*

count=1;
nstates=maxarray(X,2);
varnum=size(X,1);
for x1=1:varnum
    for x2 = x1+1:varnum
        edge(count,:)=[x1 x2];
        var(count)=MIemp(X(x1,:),X(x2,:),nstates(x1),nstates(x2));
        count=count+1;
    end
end

[xxx b]=sort(var,'descend');
edgelist=edge(b(1:count-1),:);

A=spantree(edgelist);
A=triu(A);
