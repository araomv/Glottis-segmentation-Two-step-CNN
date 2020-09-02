foldfile = {'val','train'}
for i=1:2
fno = foldfile{i}
D1 = ['./Data/NO/CNN1/image' fno '/'];
Imgs = dir(fullfile(D1,'*.png'));
for j=1:length(Imgs)
        name1 = Imgs(j).name
        file1 = ['./Data/NO/CNN1/anno' fno '/']  ; %pred
        file2 = ['./Data/NO/CNN1/image' fno '/'] ; 
        file3 = ['./Data/WO/CNN1/image' fno '/']; %output image
        file4 = ['./Data/WO/CNN1/anno' fno '/'];
        f1 = [file1 name1]
        f2 = [file2 name1]
        in1 = imread(f1);
        im1 = imread(f2);
        in2 = in1(:,:,1);
    
        gt = regionprops(in2,'Orientation');
        angle=gt.Orientation
        angle = round(angle)
        if angle < 0
            angle = 180+angle
        end
        name2 = strtok(name1,'.')
    for k=60-angle:120-angle
        w = imrotate(in1,k,'crop');
        w1 = imrotate(im1,k,'crop');
        %name = ['C:\Users\Varun\Desktop\dataaug\' 'pic' num2str(k) '.png']
        f3 = [file3 name2 '__' num2str(k) '.png']
        f4 = [file4 name2 '__' num2str(k) '.png']
        imwrite(uint8(w1),f3);
        imwrite(uint8(w),f4);
    end
end

end


