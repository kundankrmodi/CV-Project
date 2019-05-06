tic
load('Item_database.mat');

total_items = 85;
items = 1 : 1 : total_items;
num_items = length(items);
match_items = cell(1, num_items);
score_items = cell(1, num_items);
num_matches = cell(1, num_items);

cart_num = 23;
threshold = 60;
test_image_RGB = imread(sprintf('cart%d.jpg', cart_num));
test_image_item_removed = test_image_RGB;
test_image_gray = rgb2gray(test_image_RGB);
[num_rows, num_cols] = size(test_image_gray);

[test_features, test_descriptors] = vl_sift(single(test_image_gray), ...
                'edgeThresh', 7, 'peakThresh', 10);

parfor item_index = 1 : num_items
    [match_items{item_index}, score_items{item_index}] = vl_ubcmatch(descriptor_db{items(item_index)}, test_descriptors);
    [u_row2, IA, ~] = unique(match_items{item_index}(2,:));
    u_row1 = match_items{item_index}(1,IA);
    match_items{item_index} = [u_row1; u_row2];
    num_matches{item_index} = size(match_items{item_index},2) ;
end

counter1 = 1;
res_matches = match_items;
res_ransac = zeros(1,length(num_matches));
for match_num = 1:length(num_matches)
    res_ransac(match_num) = num_matches{match_num};
end
[ransac_sorted, ransac_sorted_index] = sort(res_ransac, 'descend');

loop_condition1 = 0;
while(loop_condition1 == 0)
       if(max(res_ransac) < threshold)
        loop_condition1 = 1;
       else
           ransac_sorted_index = items(ransac_sorted_index);
           res_ransac = zeros(1, length(items));
           rem_items_sorted = sort(ransac_sorted_index, 'ascend');
           for match_num = 1:length(items)
               res_ransac(match_num) = length(res_matches{match_num});
           end
           [ransac_sorted,ransac_sorted_index] = sort(res_ransac, 'descend');
           ransac_sorted_index = ransac_sorted_index(1:length(rem_items_sorted));
           
           counter2 = 1;
           loop_condition2 = 1;
           while(loop_condition2 == 1)
               X1 = feature_db{ransac_sorted_index(counter2)}(1:2, match_items{ransac_sorted_index(counter2)}(1,:)) ; X1(3,:) = 1 ;
               X2 = test_features(1:2,match_items{ransac_sorted_index(counter2)}(2,:)) ; X2(3,:) = 1 ;
               clear H score ok;
               radius = 6;
               for counter3 = 1:100
                   subset = vl_colsubset(1:res_ransac(ransac_sorted_index(counter2)), 4) ;
                   M = [] ;
                   for z = subset
                       M = cat(1, M, kron(X1(:,z)', vl_hat(X2(:,z)))) ;
                   end
                   [U,S,V] = svd(M) ;
                   H{counter3} = reshape(V(:,9),3,3) ;
                   X2_ = H{counter3} * X1 ;
                   du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
                   dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
                   ok{counter3} = (du.*du + dv.*dv) < radius*radius ;
                   homography_score(counter3) = sum(ok{counter3}) ;
               end
               [homography_score, best] = max(homography_score) ;
               H = H{best} ;
               ok = ok{best} ;
               Template = H;
               item_index = Template * [bounding_box_db{ransac_sorted_index(counter2)}; ones(1, size(bounding_box_db{ransac_sorted_index(counter2)},2))];
               p = item_index(3,:);
               boundingBoxTransformed = [item_index(1,:)./p; item_index(2,:)./p];
               mask = ~poly2mask(boundingBoxTransformed(1,:), boundingBoxTransformed(2,:), num_rows, num_cols);
               maskInfo = regionprops(~mask, 'Centroid');
               loop_condition2 = 0;
               if(length(maskInfo) ~= 1)
                   loop_condition2 = 1;
                   counter2 = counter2+1;
               end
           end
           
           pos_centroid(1,counter1) = round(maskInfo.Centroid(1));
           pos_centroid(2,counter1) = round(maskInfo.Centroid(2));
           price_of_items(counter1) = price_db{ransac_sorted_index(counter2)};
           name_of_items{counter1} = name_db{ransac_sorted_index(counter2)};
           
           for color = 1 : 3
                test_image_item_removed(:,:,color) = test_image_item_removed(:,:,color).*uint8(mask);
           end
           
           res_matches{ransac_sorted_index(counter2)} = [0;0];
           ransac_sorted_index = ransac_sorted_index(counter2+1:end);
           
           for item_index = ransac_sorted_index
               for match_num = 1:size(res_matches{item_index}, 2)
                   index_query_feature =  res_matches{item_index}(2,match_num);
                   if(index_query_feature == 0 && numel(res_matches{item_index}) == 2)
                       break
                   elseif(index_query_feature > 0)
                       query_feature = test_features(:, index_query_feature);
                       X_query_feature = round(query_feature(1));
                       Y_query_feature = round(query_feature(2));
                       if(mask(Y_query_feature, X_query_feature) == 0)
                           res_matches{item_index}(:,match_num) = 0;
                           res_ransac(item_index) = res_ransac(item_index) - 1;  
                       end
                   end
               end
               [u_row2, IA, ~] = unique(res_matches{item_index}(2,:));
               u_row1 =  res_matches{item_index}(1,IA);
               res_matches{item_index} = [u_row1; u_row2];
           end
       end
    if(loop_condition1 == 0)
        counter1 = counter1+1;
    end
end

figure(cart_num), 
imshow(test_image_RGB); 
hold on;
for item_index = 1 : length(price_of_items)
    textborder(pos_centroid(1,item_index)-50, pos_centroid(2,item_index), ...
                    ['$',num2str(price_of_items(item_index))],'g', 'w', 'FontSize', 30);
    textborder(pos_centroid(1,item_index)-50, pos_centroid(2,item_index) - 115, ...
                    name_of_items{item_index},'g', 'w', 'FontSize', 30);
end

textborder(50, 50, ['Total: $',num2str(sum(price_of_items))],'g', 'w', 'FontSize', 40);
drawnow;
hold off;
toc
saveas(gcf, sprintf('Result-cart%d.jpg', cart_num));