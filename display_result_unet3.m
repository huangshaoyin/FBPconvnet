


test_elips=load('test_elips.mat');
test_label=double(test_elips.label);
test_sparse=double(test_elips.sparse);


unet3=load('unet3.mat');
unet3=double(unet3.unet3);

 num_picture=512;
  
 num=25;
 test=zeros(num_picture,num_picture);
 test(:,:)=unet3(num,:,:);


mask=zeros(num_picture,num_picture);

mask(:,:)=test_label(:,:,1,num);

test_input=zeros(num_picture,num_picture);

test_input(:,:)=test_sparse(:,:,1,num);


rmse=sum(sum(mask.^2))/sum(sum((mask-test).^2));

figure,
imagesc(xpart,ypart,test);title('Ԥ��ͼƬ');
figure,
imagesc(xpart,ypart,mask);title('�ο�ͼƬ');
figure,
imagesc(xpart,ypart,test_input),title('����ͼƬ');























