close all
clear 
clc

N = 1e4;
load chirp; s1=y(1:N);
load gong;  s2=y(1:N);

s1=s1-mean(s1); s1=s1/std(s1);
s2=s2-mean(s2); s2=s2/std(s2);

% Combine sources into vector variables.
s=[s1,s2];

A = randn(2,2);

% generate the mixing matrix
x = s*A;

% Preprocessing step for ICA
% Centering the data (zero mean)
xm=repmat(mean(x),size(x,1),1);
xnew = x-xm;   

% Whitening transformation
[E,c]=eig(cov(xnew));
sq=inv(sqrtm(c)); % inverse of square root
xx=[E*sq*E'*xnew']';
z = xx;


M = 2; % Number of components to be identified
maxiter = 100;

for jj = 1:M
    % Initialise unmixing vector to random vector 
    W = randn(1,M);
    % Normalize
    W=W/norm(W);

    % Initialise y, the estimated source signal.
    y = z*W';


    for ii = 1:maxiter
        % Deflations for determining each independent component
        if(jj == 2)
            W1 = wn(1,:);
            % Gram-Schmidt 
            W = W - (W*W1')*W1;
            W = W/norm(W);
        end
        y = z*W';
        % Kurtosis
        K = mean(y.^4)-3; 

        y3 = y.^3;
        yy3 = repmat(y3,1,2);
        % Derivative of Kurtosis
        g = mean(yy3.*z);
            
        % Update W - Fixed Point Algorithm
        W = g;
%         W = g-3*W;
%         W = W+.1*g;
        % Renormalize W
        W = W/norm(W);
    
    end
    wn(jj,:) = W;
    % Store the identified component    
    yn(jj,:) = y;
end
% Plot results
figure;subplot(2,1,1);plot(s(:,1));title('Original Signals');subplot(2,1,2);plot(s(:,2));
figure;subplot(2,1,1);plot(x(:,1));title('Mixed Signals');subplot(2,1,2);plot(x(:,2));
figure;subplot(2,1,1);plot(yn(1,:));title('Identified Components');subplot(2,1,2);plot(yn(2,:));

% Play sounds
soundsc(yn(1,:))
soundsc(yn(2,:))