function img_section(PATH, img_number, row_divide, col_divide)

    for(i = 1:img_number)
    
        img = imread([PATH '\P (' int2str(i) ').jpg']);
%         img = imread([PATH '\P (3).jpg']);
%         img = 0.46*imnoise(img,'gaussian',0.07,0.0003);
        if(length(size(img)) == 3)
                    img = rgb2gray(img);
        end
        [m n] = size(img);
        row_N = round(m/row_divide);
        col_N = round(n/col_divide);
        cnt = 1;
        ii = 2;
            for(jj = 1:col_divide)
                if(cnt~=  3) && cnt~=5   
                imwrite(img( (ii-1)*row_N+1 : (ii*row_N) , (jj-1)*col_N+1 : (jj*col_N)  ), [PATH(1:end) '\Aphtae\N (' int2str(i) ')_(' int2str(cnt) ')' int2str(col_divide) '.bmp']);%-12
                else
                imwrite(img( (ii-1)*row_N+1 : (ii*row_N) , (jj-1)*col_N+1 : (jj*col_N)  ), [PATH(1:end) '\Aphtae\P (' int2str(i) ')_(' int2str(cnt) ')' int2str(col_divide) '.bmp']);%-12
                end
                cnt = cnt+1;                           
            end
        
        
    end


end