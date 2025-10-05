clc;
clear;

%% IMPORT (Expe.) DATA
% Import Torso States:
     Import1 = readtable('PosTorso0.txt','ReadVariableNames',false); 
     Torso_x = table2array(Import1,1);
     Import2 = readtable('PosTorso7.txt','ReadVariableNames',false); 
     Torso_v = table2array(Import2,1);
     Import3 = readtable('PosTorso1.txt','ReadVariableNames',false);  
 Torso_Pitch = table2array(Import3,1);

% Import Joints States:
     Import4 = readtable('q0.txt','ReadVariableNames',false);    
      j(1,:) = table2array(Import4,1);
     Import5 = readtable('q1.txt','ReadVariableNames',false);  
      j(2,:) = table2array(Import5,1);    
     Import6 = readtable('q2.txt','ReadVariableNames',false);    
      j(3,:) = table2array(Import6,1);     
     Import7 = readtable('q3.txt','ReadVariableNames',false);  
      j(4,:) = table2array(Import7,1);
     Import8 = readtable('q4.txt','ReadVariableNames',false); 
      j(5,:) = table2array(Import8,1);
     Import9 = readtable('q5.txt','ReadVariableNames',false); 
      j(6,:) = table2array(Import9,1);
    Import10 = readtable('q6.txt','ReadVariableNames',false);
      j(7,:) = table2array(Import10,1);
    Import11 = readtable('q7.txt','ReadVariableNames',false); 
      j(8,:) = table2array(Import11,1);
    Import12 = readtable('q8.txt','ReadVariableNames',false); 
      j(9,:) = table2array(Import12,1); 
    Import13 = readtable('q9.txt','ReadVariableNames',false);
     j(10,:) = table2array(Import13,1);
    Import14 = readtable('q10.txt','ReadVariableNames',false); 
     j(11,:) = table2array(Import14,1);
    Import15 = readtable('q11.txt','ReadVariableNames',false); 
     j(12,:) = table2array(Import15,1); 

     Import16 = readtable('dq0.txt','ReadVariableNames',false);    
      dj(1,:) = table2array(Import16,1);
     Import17 = readtable('dq1.txt','ReadVariableNames',false);  
      dj(2,:) = table2array(Import17,1);
     Import18 = readtable('dq2.txt','ReadVariableNames',false);   
      dj(3,:) = table2array(Import18,1);
     Import19 = readtable('dq3.txt','ReadVariableNames',false);  
      dj(4,:) = table2array(Import19,1);
     Import20 = readtable('dq4.txt','ReadVariableNames',false); 
      dj(5,:) = table2array(Import20,1);
     Import21 = readtable('dq5.txt','ReadVariableNames',false); 
      dj(6,:) = table2array(Import21,1);
     Import22 = readtable('dq6.txt','ReadVariableNames',false);
      dj(7,:) = table2array(Import22,1);
     Import23 = readtable('dq7.txt','ReadVariableNames',false); 
      dj(8,:) = table2array(Import23,1);
     Import24 = readtable('dq8.txt','ReadVariableNames',false); 
      dj(9,:) = table2array(Import24,1);
     Import25 = readtable('dq9.txt','ReadVariableNames',false);
     dj(10,:) = table2array(Import25,1);
     Import26 = readtable('dq10.txt','ReadVariableNames',false); 
     dj(11,:) = table2array(Import26,1);
     Import27 = readtable('dq11.txt','ReadVariableNames',false); 
     dj(12,:) = table2array(Import27,1); 
  
% Import Torques:
     Import28 = readtable('tauM0.txt','ReadVariableNames',false);    
     tau(1,:) = table2array(Import28,1);
     Import29 = readtable('tauM1.txt','ReadVariableNames',false);  
     tau(2,:) = table2array(Import29,1);
     Import30 = readtable('tauM2.txt','ReadVariableNames',false);   
     tau(3,:) = table2array(Import30,1);
     Import31 = readtable('tauM3.txt','ReadVariableNames',false);  
     tau(4,:) = table2array(Import31,1);
     Import32 = readtable('tauM4.txt','ReadVariableNames',false); 
     tau(5,:) = table2array(Import32,1);
     Import33 = readtable('tauM5.txt','ReadVariableNames',false); 
     tau(6,:) = table2array(Import33,1);
     Import34 = readtable('tauM6.txt','ReadVariableNames',false);
     tau(7,:) = table2array(Import34,1);
     Import35 = readtable('tauM7.txt','ReadVariableNames',false); 
     tau(8,:) = table2array(Import35,1);
     Import36 = readtable('tauM8.txt','ReadVariableNames',false); 
     tau(9,:) = table2array(Import36,1);
     Import37 = readtable('tauM9.txt','ReadVariableNames',false);
    tau(10,:) = table2array(Import37,1);
     Import38 = readtable('tauM10.txt','ReadVariableNames',false); 
    tau(11,:) = table2array(Import38,1);
     Import39 = readtable('tauM11.txt','ReadVariableNames',false); 
    tau(12,:) = table2array(Import39,1);

% Import Forces (Calc.):
    Import40 = readtable('forceFeetGlobal0.txt','ReadVariableNames',false); 
    FRf(1,:) = table2array(Import40,1);
    Import41 = readtable('forceFeetGlobal1.txt','ReadVariableNames',false);
    FRf(2,:) = table2array(Import41,1);
    Import42 = readtable('forceFeetGlobal2.txt','ReadVariableNames',false);
    FRf(3,:) = table2array(Import42,1);
     
    Import46 = readtable('forceFeetGlobal6.txt','ReadVariableNames',false); 
    RRf(1,:) = table2array(Import46,1);
    Import47 = readtable('forceFeetGlobal7.txt','ReadVariableNames',false); 
    RRf(2,:) = table2array(Import47,1);
    Import48 = readtable('forceFeetGlobal8.txt','ReadVariableNames',false);  
    RRf(3,:) = table2array(Import48,1);
    
% Import Forces (Exp.):
    Import43 = readtable('simforceFeetGlobal0.txt','ReadVariableNames',false);        
    Import44 = readtable('simforceFeetGlobal1.txt','ReadVariableNames',false);   
    Import45 = readtable('simforceFeetGlobal2.txt','ReadVariableNames',false);    
    sFRf(1,:) = table2array(Import45,1);
    
    Import49 = readtable('simforceFeetGlobal6.txt','ReadVariableNames',false);        
    Import50 = readtable('simforceFeetGlobal7.txt','ReadVariableNames',false);    
    Import51 = readtable('simforceFeetGlobal8.txt','ReadVariableNames',false);  
    sRRf(1,:) = table2array(Import51,1);
        
% Import Time:
        Import52 = readtable('desPosTorso9.txt','ReadVariableNames',false);   
FrontStance(1,:) = table2array(Import52)';
        Import53 = readtable('desPosTorso10.txt','ReadVariableNames',false); 
 RearStance(1,:) = table2array(Import53)';

% Work:
 P = tau.*dj;
dt = 0.002; % 500 Hz
 W = tau.*dj.*dt; % incremental work
pW = W;
nW = W;
pW(pW < 0) = 0;
nW(nW > 0) = 0;

%% Actual Contact States

% fAS = sFRf;
% fAS(fAS > 0) = 1;
% 
% rAS = sRRf;
% rAS(rAS > 0) = 1;
% 
%  for i = 2 : size(Import51,1)
%     FLAS(1,i-1) = fAS(1,i) - fAS(1,i-1);
%     RLAS(1,i-1) = rAS(1,i) - rAS(1,i-1);
%  end
% 
% AFL = find(FLAS(1,:) == -1);
% AFT = find(FLAS(1,:) ==  1);
% 
% ARL = find(RLAS(1,:)== -1);
% ART = find(RLAS(1,:)==  1);
% 
% %%
% 
% I = find(AFL == 7699);
% F = I+1;
% 
% fAS(AFL(I)+1:AFL(F))
% rAS(AFL(I)+1:AFL(F))
% 
% 
% 
% figure
% plot([AFL;AFL],[1;-1],'-b',[AFT;AFT],[1;-1],'-.b',[ARL;ARL],[1;-1],'-r',[ART;ART],[1;-1],'-.r');

%% Contact States

E_Time = 0:0.002:((size(Import1,1))*0.002)-0.002;

FrontStance(1,:) = FrontStance(1,:) == 5;
RearStance(1,:) =  RearStance(1,:) == 5;
SF = FrontStance(1,:);
SR = RearStance(1,:);
                 
 for i = 2 : size(Import52,1)
    FrontStance(1,i-1) = FrontStance(1,i) - FrontStance(1,i-1);
     RearStance(1,i-1) = RearStance(1,i)  - RearStance(1,i-1);
 end

% In blue front leg stance,
F_L = find(FrontStance(1,:) == -1);
F_T = find(FrontStance(1,:) ==  1);
F_L_t = F_L * 0.002;
F_T_t = F_T * 0.002;

% In red rear leg stance,
R_L = find(RearStance(1,:)== -1);
R_T = find(RearStance(1,:)==  1);
R_L_t = R_L * 0.002;
R_T_t = R_T * 0.002;

figure
plot([F_L;F_L],[1;-1],'-b',[F_T;F_T],[1;-1],'-.b',[R_L;R_L],[1;-1],'-r',[R_T;R_T],[1;-1],'-.r');

%%

% figure 
% plot([F_L;F_L],[1;-1],'-b');
% hold on
% plot([F_T;F_T],[1;-1],'-.b');
% hold on
% plot([R_L;R_L],[1;-1],'-r');
% hold on
% plot([R_T;R_T],[1;-1],'-.r');

% figure
% plot([F_L F_T R_L R_T; F_L F_T R_L R_T], [1 1 1 1; -1 -1 -1 -1], {'-b','-.b','-r','-.r'});

% figure;
% a1 = gca;
% for i = 1 : length(F_L)
%     plot([F_L(i);F_L(i)],[1;-1],'-b');
%     hold on
% end
% % xlim([10876 11512])
% 
% for i = 1 : length(F_T)
%     plot([F_T(i);F_T(i)],[1;-1],'-.b');
%     hold on
% end
% % xlim([10876 11512])

% for i = 1 : length(R_L)
%     plot([R_L(i);R_L(i)],[1;-1],'-r');
%     hold on
% end
% % xlim([10876 11512])
% 
% for i = 1 : length(R_T)
%     plot([R_T(i);R_T(i)],[1;-1],'-.r');
%     hold on
% end
% % xlim([10876 11512])

% copyStateContactFig = gca;

%% Plot Data
sample = (1:length(dj(1,:)));
    t_ = (1:length(dj(1,:)))*dt; % we care about delt T 
   t_0 = [0, t_(1:end-1)]; % we care about delt T 
   
     T = sample;
    
% FR Leg
% Velocity
figure
tiledlayout(3,3); 
nexttile
plot(T,dj(1,:)); title('FR Hip'); xlabel('t'); ylabel('rad/s');
nexttile
plot(T,dj(2,:)); title('FR Thigh'); xlabel('t'); ylabel('rad/s');
nexttile
plot(T,dj(3,:)); title('FR Calf'); xlabel('t'); ylabel('rad/s');


% Torque
nexttile
plot(T,tau(1,:)); title('FR Hip'); xlabel('t'); ylabel('Nm');
nexttile
plot(T,tau(2,:)); title('FR Thigh'); xlabel('t'); ylabel('Nm');
nexttile
plot(T,tau(3,:)); title('FR Calf'); xlabel('t'); ylabel('Nm');

% Power
nexttile
plot(T,P(1,:)); title('FR Hip'); xlabel('t'); ylabel('Watt');
nexttile
plot(T,P(2,:)); title('FR Thigh'); xlabel('t'); ylabel('Watt');
nexttile
plot(T,P(3,:)); title('FR Calf'); xlabel('t'); ylabel('Watt');


% RR Leg
% Velocity
figure
tiledlayout(3,3);
nexttile
plot(T,dj(7,:)); title('RR Hip'); xlabel('t'); ylabel('rad/s');
nexttile
plot(T,dj(8,:)); title('RR Thigh'); xlabel('t'); ylabel('rad/s');
nexttile
plot(T,dj(9,:)); title('RR Calf'); xlabel('t'); ylabel('rad/s');

% Torque
nexttile
plot(T,tau(7,:)); title('RR Hip'); xlabel('t'); ylabel('Nm');
nexttile
plot(T,tau(8,:)); title('RR Thigh'); xlabel('t'); ylabel('Nm');
nexttile
plot(T,tau(9,:)); title('RR Calf'); xlabel('t'); ylabel('Nm');

% Power
nexttile
plot(T,P(7,:)); title('RR Hip'); xlabel('t'); ylabel('Watt');
nexttile
plot(T,P(8,:)); title('RR Thigh'); xlabel('t'); ylabel('Watt');
nexttile
plot(T,P(9,:)); title('RR Calf'); xlabel('t'); ylabel('Watt');

%% Plot Several Strides
range = 7226:8490;

% FR Leg
% Velocity
figure
tiledlayout(3,3); 
nexttile
plot(T(range),dj(1,range)); title('FR Hip'); xlabel('t'); ylabel('rad/s');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),dj(2,range)); title('FR Thigh'); xlabel('t'); ylabel('rad/s');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),dj(3,range)); title('FR Calf'); xlabel('t'); ylabel('rad/s');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);

% Torque
nexttile
plot(T(range),tau(1,range)); title('FR Hip'); xlabel('t'); ylabel('Nm');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),tau(2,range)); title('FR Thigh'); xlabel('t'); ylabel('Nm');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),tau(3,range)); title('FR Calf'); xlabel('t'); ylabel('Nm');
hold on 
plot([F_L;F_L],[20;-20],'-k',[F_T;F_T],[20;-20],'-.k');
xlim([range(1) range(end)]);

% Power
nexttile
plot(T(range),P(1,range)); title('FR Hip'); xlabel('t'); ylabel('Watt');
hold on 
plot([F_L;F_L],[60;-60],'-k',[F_T;F_T],[60;-60],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),P(2,range)); title('FR Thigh'); xlabel('t'); ylabel('Watt');
hold on 
plot([F_L;F_L],[60;-60],'-k',[F_T;F_T],[60;-60],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),P(3,range)); title('FR Calf'); xlabel('t'); ylabel('Watt');
hold on 
plot([F_L;F_L],[60;-60],'-k',[F_T;F_T],[60;-60],'-.k');
xlim([range(1) range(end)]);


% RR Leg
% Velocity
figure
tiledlayout(3,3);
nexttile
plot(T(range),dj(7,range)); title('RR Hip'); xlabel('t'); ylabel('rad/s');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),dj(8,range)); title('RR Thigh'); xlabel('t'); ylabel('rad/s');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),dj(9,range)); title('RR Calf'); xlabel('t'); ylabel('rad/s');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);

% Torque
nexttile
plot(T(range),tau(7,range)); title('RR Hip'); xlabel('t'); ylabel('Nm');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),tau(8,range)); title('RR Thigh'); xlabel('t'); ylabel('Nm');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),tau(9,range)); title('RR Calf'); xlabel('t'); ylabel('Nm');
hold on 
plot([R_L;R_L],[20;-20],'-k',[R_T;R_T],[20;-20],'-.k');
xlim([range(1) range(end)]);

% Power
nexttile
plot(T(range),P(7,range)); title('RR Hip'); xlabel('t'); ylabel('Watt');
hold on 
plot([R_L;R_L],[60;-60],'-k',[R_T;R_T],[60;-60],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),P(8,range)); title('RR Thigh'); xlabel('t'); ylabel('Watt');
hold on 
plot([R_L;R_L],[60;-60],'-k',[R_T;R_T],[60;-60],'-.k');
xlim([range(1) range(end)]);
nexttile
plot(T(range),P(9,range)); title('RR Calf'); xlabel('t'); ylabel('Watt');
hold on 
plot([R_L;R_L],[60;-60],'-k',[R_T;R_T],[60;-60],'-.k');
xlim([range(1) range(end)]);

% Plot GRF

figure
tiledlayout(2,2);

% Tile 1
nexttile
plot(T,FRf(1,:),'g')
hold on
plot(T,FRf(2,:),'r')
hold on
plot(T,FRf(3,:),'b')
hold on
plot([F_L;F_L],[100;-100],'-k',[F_T;F_T],[100;-100],'-.k');
xlim([range(1) range(end)]);
ylim([-250 100])
yline(-100,'--');
yline(-150,'--');
yline(-200,'--');
title('Front GRF (Expe.)');

% Tile 2
nexttile
plot(T,RRf(1,:),'g')
hold on
plot(T,RRf(2,:),'r')
hold on
plot(T,RRf(3,:),'b')
hold on
plot([R_L;R_L],[100;-100],'-k',[R_T;R_T],[100;-100],'-.k');
xlim([range(1) range(end)]);
ylim([-250 100])
yline(-100,'--');
yline(-150,'--');
yline(-200,'--');
title('Rear GRF (Expe.)');

% Tile 4
nexttile
plot(T,sFRf(1,:))
hold on
plot([F_L;F_L],[300;-300],'-k',[F_T;F_T],[300;-300],'-.k');
xlim([range(1) range(end)]);
ylim([0 300])
yline(100,'--');
yline(150,'--');
yline(200,'--');
title('Front GRF (Sim.)');

% Tile 4
nexttile
plot(T,sRRf(1,:))
hold on
plot([R_L;R_L],[300;-300],'-k',[R_T;R_T],[300;-300],'-.k');
xlim([range(1) range(end)]);
ylim([0 300])
yline(100,'--');
yline(150,'--');
yline(200,'--');
title('Rear GRF (Sim.)');

%% Calculate CoT

% A1 Physical Parameters 
      mass = [4.713, 0.001, 0.696*4, 1.013*4, 0.166*4, 0.06*4]; % Masses of [Torso, IMU, Hip, Thigh, Calf, Foot]
      Mass = sum(mass); 
         g = 9.81;          
        dt = 0.002;

        Cot_C = [];
        
%figure    
 %for pp = 10 : min([length(F_T),length(F_L),length(R_T),length(R_L)])
 for pp = 40 : 80
         I = pp;
         F = I+1;        
       x_i = Torso_x(F_L(I)+1);    
       x_f = Torso_x(F_L(F)); 
  Velocity = (x_f - x_i)/ (E_Time(F_L(F))-E_Time(F_L(I)+1));
  
% COT 
%     if Velocity>0.0 && Velocity<5
%            S = [repmat(SF(F_L(I)+1:F_L(F)),[6 1]);repmat(SR(F_L(I)+1:F_L(F)),[6 1])];
    
         COT = sum(sum(abs(tau(:,F_L(I)+1:F_L(F)).*dj(:,F_L(I)+1:F_L(F)).*dt)))/abs((Mass * g * (x_f - x_i)))  %S.*
         Cot_C(end+1) = COT;
    
        if COT<20
              p = plot(abs(Velocity),COT,'ob');
              p.MarkerFaceColor = [0 0 1];
              xlim([0 1])
              hold on
              %ylim([0 2])
            %  ylim([0 4])
        end
 end
 
%  writematrix(Cot_C', 'Cot_C34.csv');
%  [~,NAME,name] = fileparts(pwd);
%            title([NAME name])

