


test_elips=load('test_elips.mat');
test_label=double(test_elips.label);
test_sparse=double(test_elips.sparse);

unet=load('unet.mat');
unet=double(unet.unet);

 num_picture=512;
  
 num=25;
 test=zeros(num_picture,num_picture);
 test(:,:)=unet(num,:,:);


mask=zeros(num_picture,num_picture);

mask(:,:)=test_label(:,:,1,num);

test_input=zeros(num_picture,num_picture);
test_input(:,:)=test_sparse(:,:,1,num);

rmse=sum(sum(mask.^2))/sum(sum((mask-test).^2));

rmse_input=sum(sum(mask.^2))/sum(sum((mask-test_input).^2));

figure,
imagesc(test);title('Ô¤²âÍ¼Æ¬');
figure,
imagesc(mask);title('²Î¿¼Í¼Æ¬');
figure,
imagesc(test_input),title('ÊäÈëÍ¼Æ¬');























