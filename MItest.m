function mi = MItest(a,b)
%culate MI of a and b in the region of the overlap part


[Ma,Na] = size(a);
[Mb,Nb] = size(b);
M=min(Ma,Mb);
N=min(Na,Nb);


hab = zeros(256,256);
ha = zeros(1,256);
hb = zeros(1,256);


if max(max(a))~=min(min(a))
    a = (a-min(min(a)))/(max(max(a))-min(min(a)));
else
    a = zeros(M,N);
end

if max(max(b))-min(min(b))
    b = (b-min(min(b)))/(max(max(b))-min(min(b)));
else
    b = zeros(M,N);
end

a = double(int16(a*255))+1;
b = double(int16(b*255))+1;


for i=1:M
    for j=1:N
       indexx =  a(i,j);
       indexy = b(i,j) ;
       hab(indexx,indexy) = hab(indexx,indexy)+1;
       ha(indexx) = ha(indexx)+1;
       hb(indexy) = hb(indexy)+1;
   end
end


hsum = sum(sum(hab));
index = find(hab~=0);
p = hab/hsum;
Hab = sum(sum(-p(index).*log(p(index))));


hsum = sum(sum(ha));
index = find(ha~=0);
p = ha/hsum;
Ha = sum(sum(-p(index).*log(p(index))));


hsum = sum(sum(hb));
index = find(hb~=0);
p = hb/hsum;
Hb = sum(sum(-p(index).*log(p(index))));


mi = Ha+Hb-Hab;


%mi = hab/(Ha+Hb);
