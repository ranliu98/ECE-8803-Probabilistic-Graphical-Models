function [marg mess FactorGraph_A]=LBP_self(phi)
import brml.*

FactorGraph_A = FactorGraph(phi);
messlidx = find(FactorGraph_A); % message indices

for loop=1:50
    r=1:20;
    for i=1:20
        FactorGraph_A(messlidx(i))=r(i);
        k(r(i))=i;
    end
    
    if loop == 1
        [marg mess]=sumprodFG(phi,FactorGraph_A);
        mess(1:20)=mess;
        mess=condpot(mess); marg=condpot(marg);
    else
        [marg mess(k)]=sumprodFG(phi,FactorGraph_A,mess(k));
        mess=condpot(mess);
        marg=condpot(marg);  
    end
end
mess=mess(k);