
 load('data.mat')

do_round = 1;

N = 10;

M= 14;


%%

load('/home/emre/Documents/kodlar/pclouds/data.mat');

T = [ones(1,122)];

w1 = double(w1);

w2 = double(w2);

b1 = double(b1);

b2 = double(b2);

if do_round

    C1 = 2^N / max([max(w1(:)) max(b1)]);

    C2 = 2^N / max([max(w2(:)) max(b2)]);

else

    C1=1;

    C2=1;

end

if (do_round)

    iw1  = round(C1*w1);

    ib1  = round(C1*b1);

    iw2  = round(C2*w2);

    ib2  = round(C2*b2);

       

else

    iw1  = w1;

    ib1  = b1;

    iw2  = w2;

    ib2  = b2;

end

o1 = (double(T)*iw1+ib1)/C1;

ro1 = o1.*(o1>0);

ro2 = round(ro1*2^M);

o2 = ro2*iw2+ib2*2^M;

o3 = o2/(C2*2^M);

probs = exp(o3)/sum(exp(o3))


% 
% do_round = 1;
% N = 40;
% 
% %%
% load('/home/emre/Documents/kodlar/pclouds/data.mat')
% 
% T = ones(1,122);
% % T = T(end:-1:1);
% w1 = double(w1);
% w2 = double(w2);
% b1 = double(b1);
% b2 = double(b2);
% 
% if do_round
%     C1 = 2^N / max([max(w1(:)) max(b1)]);
%     C2 = 2^N / max([max(w2(:)) max(b2)]);
% else
%     C1=1; 
%     C2=1;
% end
% 
% if (do_round)
%     iw1  = round(C1*w1);
%     ib1  = round(C1*b1);
%     iw2  = round(C2*w2);
%     ib2  = round(C2*b2);
% 
% else
%     iw1  = w1;
%     ib1  = b1;
%     iw2  = w2;
%     ib2  = b2;
% end
% 
% 
% 
% o1 = double(T)*iw1+ib1;
% ro1 = o1.*(o1>0);
% 
% o2 = ro1*iw2+ib2;
% 
% o3 = o2/(C1*C2);
% 
% probs = exp(o3)/sum(exp(o3))
% 
% 
% 
% 
% 
% % o1 = double(T)*iw1+ib1;
% 
% 
% 
% 
% 
% % Input = ones(123,1);
% % 
% % Weights = rand(123,244);
% % 
% % outp1 = Weights*input;
% % 
% % outp1(i1) = sum( Weights(i1,:)*input); % 1 + 2^(-20)-1  1 -1 + 2^(-20)
% % 
% % outp2 = h(outp1);
% % 
% % Weights2 = rand(244,2);
% % 
% % outp3 = Weights2*output2;
% % 
% % [Prob 1-Prob] = h2(outp2);
% % 
% % Count -> round(Prob*2^10)
% % 
% % norm = max(Weights);
% % 
% % IntegerWeights = round(Weights/norm*2^32);
% % 
% % norm1 = 2^32/norm  %
% % 
% % Int_outp1 = IntegerWeights*input;
% % 
% % Aprrox_outp2 = h(Int_outp1/norm1)
% % 
% % Aprrox_outp2 = round(Aprrox_outp2/max(Aprrox_outp2)*2^32);