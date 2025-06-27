clear;

file_path = 'D:\Face_IQA\code\FAS_DRL\dataset\REPLAY_test\test_first_face_dlib_zl\';% 图像文件夹路径  
img_path_list = dir(strcat(file_path,'*png'));%获取该文件夹中所有tif格式的图像
img_num = length(img_path_list);%获取图像总数量
I=cell(1,img_num);
feature=[];
if img_num > 0 %有满足条件的图像  
    for j = 1:img_num %逐一读取图像
        image_name = img_path_list(j).name;% 图像名
        image = imread(strcat(file_path,image_name));
        image = imresize(image, [256,256]);
        fprintf('%d %s\n',j,strcat(file_path,image_name));% 显示正在处理的图像名
        feat = biqi_statistics(image);
        feature = [feature; feat];
        % fprintf('%f \n',j,qualityscore(j));
    end
end