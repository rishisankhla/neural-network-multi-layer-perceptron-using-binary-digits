%IS53002B: Coursework Assignment
%student Name – Rishi Sankhla
%Student No. – 33724434

%initialisation of data and setting parameters
y=[1,0];
x1=[0,1];
x2=[1,0];
n=0.2;
our_Weights=[0.2,0.1,0.2,-0.1,-0.2,0.1,-0.1,0.2,-0.2];
hidden_weights_initial = zeros(2,3);
hidden_xvalues_initial = zeros(2,3);
n_hidden = 3;
n_input = 2;

%displaying initial weights
disp("-----------initial weights-------------");
disp(our_Weights);


    
%for loop to iteerate over the length of dataset, 
%i.e 2 iteeration for 2 test cases
for i=1:size(x1,2)
    
    %calcualting output of hidden nodes
    hidden_weights_initial = adjustweights(hidden_weights_initial,our_Weights);
    hidden_xvalues_initial = adjustinput(hidden_xvalues_initial,i,x1,x2);
    final_hidden_output=[];

    %for loop for forward propagation
    for f=1:n_hidden
        c=0;
        for g=1:n_input
            total = hidden_weights_initial(g,f)*hidden_xvalues_initial(g,f);
            c=c+total;
        end
        final_hidden_output=[final_hidden_output,summation_sigmoid(c)];
    end
    
    %calculating final out
    Out = summation([our_Weights(9),our_Weights(2),our_Weights(8),our_Weights(5),our_Weights(7)], ...
                    [final_hidden_output(1);x1(i);final_hidden_output(2);x2(i);final_hidden_output(3)]);

    %calculating error
    beta_out = y(i)-Out;
    
    %calculating weight change, which connected to "out" node
    delta_w9_Out = n*beta_out*final_hidden_output(1); 
    delta_w2_Out = n*beta_out*x1(i);
    delta_w8_Out = n*beta_out*final_hidden_output(2); 
    delta_w5_Out = n*beta_out*x2(i); 
    delta_w7_Out = n*beta_out*final_hidden_output(3);

    %calculating beta of hidden nodes
    out_values = final_hidden_output;
    respected_weights = [our_Weights(9),our_Weights(8),our_Weights(7)];
    beta_val = [];

    for p=1:size(respected_weights,2) %for loop to calculate beta
        beta_val=[beta_val,calculate_beta(out_values(p),respected_weights(p),beta_out)];
    end
    
    %calculating weight change, which connected to "hidden" node
    delta_w1_Y1 = n*beta_val(1)*x1(i);
    delta_w3_Y2 = n*beta_val(2)*x1(i);
    delta_w4_Y2 = n*beta_val(2)*x2(i);
    delta_w6_Y5 = n*beta_val(3)*x2(i);
    
    delta_Weights=[delta_w1_Y1,delta_w2_Out,delta_w3_Y2,delta_w4_Y2,delta_w5_Out,delta_w6_Y5,delta_w7_Out,delta_w8_Out,delta_w9_Out];
    our_Weights=our_Weights+delta_Weights; %updating weights after each iteration
    
    %printing out weights and error
    disp("----------------error------------------");
    disp(beta_out);
    disp("-----------updated weights-------------");
    disp(our_Weights);
    
end


%function to calculate sigmoid
function sig = summation_sigmoid(c)
    sig = 1.0 ./( 1.0 + exp( -(c) ));
end

%function to perform summation
function sum = summation(x,y)
    sum = x*y;
end

%function to calculate hidden node beta
function val = calculate_beta(x,y,z)
    val = x*(1-x)*z*y;
end

%function to add weights
function w = adjustweights(x,our_Weights)
    x(1,1)=our_Weights(1);
    x(1,2)=our_Weights(3);
    x(2,2)=our_Weights(4);
    x(1,3)=our_Weights(6);
    w=x;
end

%function to add x_values
function out = adjustinput(x,i,x1,x2)
    x(1,1)=x1(i);
    x(1,2)=x1(i);
    x(2,2)=x2(i);
    x(1,3)=x2(i);
    out=x;
end
