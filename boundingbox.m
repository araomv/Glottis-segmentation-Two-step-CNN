folders = {'train', 'val','test'}
    for  fno = 1:3
        foldername = folders{fno}
        
        D1 = ['./Data/NO/CNN1/image' foldername '/']
        Imgs1 = dir(fullfile(D1,'*.png'));
        
       
        for j=1:length(Imgs1)
                 name1 = Imgs1(j).name
                 file1 = ['./Data/NO/CNN1/pred' foldername '/']  ; %pred CNN1
                 file2 = ['./Data/NO/CNN1/image' foldername '/'] ;  %CNN1 input image
            
                 file3 = ['./Data/NO/CNN2/image' foldername '/'] ;  %CNN1 output image
                 file4 = ['./Data/NO/CNN2/anno' foldername '/'] ;   %CNN1 output ground
                 file5 = ['./Data/NO/CNN1/anno' foldername '/'] ;    %CNN1 input ground
                
                %sav = [file3 name]
                f1 = [file1 name1]
                f2 = [file2 name1]
                f3 = [file3 name1]
                f4 = [file4 name1]
                f5 = [file5 name1]
                in1 = imread(f1);
                im1 = imread(f2);
                ip1 = imread(f5);
                %im2 = imresize(im1,[224 224]);
   
        
                in2 = imresize(in1,[576 720]);
                im2 = imresize(im1,[576 720]);
                ip2 = imresize(ip1,[576 720]);
                ip3 = ip2(:,:,1);
                dum = zeros(576,720);
        
                in3 = in2(:,:,1);
                im3 = im2(:,:,1);
        
                [x1,y1] = bwboundaries(in3);
                [a b] = size(x1);
                 area=0;
                 for i=1:a
                         y = x1{i}(:,1);
                         x = x1{i}(:,2);
                         temp_area = polyarea(x,y);
                        if area<temp_area
                            area=temp_area;
                            x_final = x;
                            y_final = y;
                        end
            
            
                 end
                final = zeros(576,720);
                fix = zeros(576,720);
                fiy = zeros(576,720);
                for i=1:576
                    fix(i,:)=i;
                end
                 for i=1:720
                  fiy(:,i)=i;
                 end
                 [in on] = inpolygon(fiy,fix,x_final,y_final);
       
                [x1,y1] = bwboundaries(in);
                x = x1{1}(:,2);
                y = x1{1}(:,1);
       
       %polyin = polyshape(x,y);
                stats = regionprops(in);
                cen = stats.Centroid
                x3 = cen(2)-112;
                y3 = cen(1)-112;
                x3 = round(x3);
                y3 = round(y3);
       
      
       
       m1 = max(x)-min(x)
       m2 = max(y)-min(y)
       
       if m1<224
            x3 = cen(2)-112;
            x3 = round(x3);
            x_start = x3
            x_end = x3+223
       else 
           x3 = round(min(x));
           x_start = x3 - 20;
           x_end = round(max(x))+20;
       end
       
       if m2<224
            y3 = cen(1)-112;
            y3 = round(y3);
            y_start = y3
            y_end = y3+223
       else 
           y3 = round(min(y));
           y_start = y3 - 20;
           y_end = round(max(y))+20;
       end
       
       if x_start < 1
           x_start = 1;
       end
       if y_start < 1
           y_start = 1;
       end
       if x_end > 576
           x_end = 576;
       end
       if y_end > 720
           y_end = 720;
       end
       
            
       
       imo = im2(x_start:x_end,y_start:y_end,:);
       ipo = ip2(x_start:x_end,y_start:y_end,:);
       imwrite(uint8(imo),f3);
       imwrite(uint8(ipo),f4);
       
       %imshow(imo)
       
       %plot(x,y)
        
        end
        
        
        
        
        
        
        
        
        
        
        
        
        
       
    end
    
