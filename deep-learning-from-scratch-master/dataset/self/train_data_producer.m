function [P N] = train_data_producer(Pnum, Nnum)


max_amp = 50;
distPATH = 'C:\Users\s9314\Desktop\deep-learning-from-scratch-master\dataset\self\testing\新增資料夾\';
row = 192;
col = 240;
sig1 = 130^2;
sig2 = 120^2;

for(i = 1:row)
    for(j = 1:col)
        gaussian_light(i,j) = 3*exp(-(  (i-row/2).^2/(2*sig1) + (j-col/2).^2/(2*sig2)       ));
    end
end

for(I=1:Pnum)
    
    model = max_amp*ones(192,240);
height = 50;
wedith = 154;
[m n] = size(model);
centery = m/2 + round(10*randn(1));
centerx = n/2+ round(20*randn(1));
thick  = round(4*rand(1)+3);

 thick_c = 5;
 thick_t = 7;
 shift_C = round(10*rand(1));
 shift_T = round(10*rand(1));
C_line = centerx - (40 +shift_C )   :   (centerx - (40 + shift_C) + thick_c);
T_line = centerx + (40 + shift_T)   :   (centerx + (40 + shift_T) + thick_t);

 cir_cneter_x_L = centerx - round(wedith/2);
 cir_cneter_x_R = centerx + round(wedith/2);
 T_amp = max_amp/(1.5*rand(1)+1.9);
 
    for(i=1:m)
        for(j=1:n)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
           if( abs(i - centery) <= (height/2) )  
                    model(i,C_line) = 10;
                    model(i,T_line) = T_amp*exp((fliplr(T_line) - min(T_line))/15) +10;
          end        
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                         
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %           
          if( abs(j - centerx) < (wedith/2+thick) )  
            if( abs(i - centery) < height/2+thick && abs(i - centery) > height/2 )
                model(i,j) = 10;
            end
          end   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %           
          if(  abs(j - centerx) >= (wedith/2+thick) )
                if(  sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_L)^2))    >   height/2  && sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_L)^2))    <   (height/2+thick)      )
                    model(i,j) = 10;
                end 
                if(  sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_R)^2))    >   height/2  && sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_R)^2))    <   (height/2+thick)      )
                    model(i,j) = 10;
                end                 
          end

        end
    end
    P(:,:,I) = uint8(gaussian_light.*model);
    P(:,:,I) = imnoise(uint8(P(:,:,I)),'gaussian',0.07,0.0003);
    imwrite(uint8(P(:,:,I)),[distPATH 'P (' int2str(I) ').bmp']);
end

for(I=1:Nnum)
    model = max_amp*ones(192,240);
height = 50;
wedith = 154;
[m n] = size(model);
centery = m/2 + round(10*randn(1));
centerx = n/2+ round(20*randn(1));
thick  = round(4*rand(1)+3);

 thick_c = 7;
 shift_C = round(15*rand(1));
C_line = centerx - (40 +shift_C )   :   (centerx - (40 + shift_C) + thick_c);

 cir_cneter_x_L = centerx - round(wedith/2);
 cir_cneter_x_R = centerx + round(wedith/2);

 
    for(i=1:m)
        for(j=1:n)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %           
          if( abs(j - centerx) < (wedith/2+thick) )  
            if( abs(i - centery) < height/2+thick && abs(i - centery) > height/2 )
                model(i,j) = 10;
            end
          end   
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %           
          if(  abs(j - centerx) >= (wedith/2+thick) )
                if(  sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_L)^2))    >   height/2  && sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_L)^2))    <   (height/2+thick)      )
                    model(i,j) = 10;
                end 
                if(  sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_R)^2))    >   height/2  && sqrt(   ((i - centery)^2) + ((j - cir_cneter_x_R)^2))    <   (height/2+thick)      )
                    model(i,j) = 10;
                end                 
          end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
           if( abs(i - centery) < (height/2+thick) )  
                    model(i,C_line) = 10;     
          end        
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
        end
    end
    N(:,:,I) = uint8(gaussian_light.*model);
    N(:,:,I) = imnoise(uint8(N(:,:,I)),'gaussian',0.07,0.0003);
    imwrite(uint8(N(:,:,I)),[distPATH 'N (' int2str(I) ').bmp']);
end



end