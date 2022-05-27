IF = [];
OCSVM = [];
RVAE = [];
CLEAN = [];
PK = [];
NOTHING = [];

for count = 0:4
    ds_name = 'wine';
    name = strcat('../data/dataPoisonSet/', ds_name, '_', num2str(count));
    load(name);
    
    name = strcat('../data/dataPoisonSet/nn/', ds_name, '_poison_', num2str(count));
    load(name);
    
    name = strcat('../data/res/train/', ds_name, '_nn_data_left_', num2str(count));
    load(name);
    
    x_mix = [x_tr; X_poison];
    y_mix = double([y_tr; y_poison]);
    
    alpha = 0.1; %Learning rate
    iter = 400; %Number of epochs
    param.M = 10; %Number of neurons
    param.alpha = alpha;
    param.nit = iter;
    param.init = 1; %Initialize weights from a given values
    param.act = 2; %Use tanh as activation function in the hidden layer
    d = size(x_tr,2);
    M = param.M;
    %Initialization of the weights
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(x_tr,y_tr,param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(x_mix,y_mix,param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    NOTHING = [NOTHING 1-error_clean_test];
    
    [net, errtr] = trainMLP(x_tr,y_tr,param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    CLEAN = [CLEAN 1-error_clean_test];
    
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(X_IF,reshape(y_IF, [], 1),param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    IF = [IF 1-error_clean_test];
    
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(X_OCSVM,reshape(y_OCSVM, [], 1),param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    OCSVM = [OCSVM 1-error_clean_test];
    
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(X_RVAE,reshape(y_RVAE,[],1),param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    RVAE = [RVAE 1-error_clean_test];
    
    w0 = randn(d+1,M).*0.1;
    w0_2 = randn(M+1,1).*0.1; 
    param.w = w0;
    param.w_2 = w0_2;
    
    [net, errtr] = trainMLP(X_PK,reshape(y_PK,[],1),param); 
    [sval,tval] = testMLP(net,x_val);
    error_clean_val = mean(tval~=y_val);
    [stst,ttst] = testMLP(net,x_tst);
    error_clean_test = mean(ttst~=y_tst);
    
    PK = [PK 1-error_clean_test];
end


fprintf('Downstream Accuracy\n');
fprintf('Noise Type: poison\n');
fprintf('Downstream Model: nn\n');
fprintf('IF: %1.4f ', mean(IF));
fprintf('OCSVM: %1.4f ', mean(OCSVM));
fprintf('RVAE: %1.4f ', mean(RVAE));
fprintf('Picket: %1.4f ', mean(PK));
fprintf('Clean: %1.4f ', mean(CLEAN));
fprintf('No Filtering: %1.4f\n', mean(NOTHING));
