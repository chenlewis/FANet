file_path = 'D:\DIQA_IF\dataset\student_card\d1_all\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*.tif'));%获取该文件夹中所有bmp格式的图像  
img_num = length(img_path_list);%获取图像总数量 
I=cell(1,img_num);
qualityscore=[];
if img_num > 0 %有满足条件的图像  
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名
        qualityscore(end+1) = brisquescore(image);
        score = qualityscore(j)
        % fprintf('%f \n',j,qualityscore(j));
    end
end
qualityscore_t = qualityscore.';