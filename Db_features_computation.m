[~, ~, data] = xlsread('Our_db.xlsx');
num_Items = length(data);
name_db = data(:,1);
price_db = data(:,2);

img_db = cell(1,numItems);
feature_db = cell(1,numItems);
descriptor_db = cell(1,numItems);
bounding_box_db = cell(1,numItems);

for item_num = 1 : num_Items
    img_db{item_num} = rgb2gray(imread(sprintf('%d.jpg', item_num)));
    [feature_db{item_num}, descriptor_db{item_num}] = vl_sift(single(img_db{item_num}), ...
                                    'edgeThresh', 7, 'peakThresh', 10);
    
    [sort_feature_X, ~] = sort(feature_db{item_num}(1,:));
    [sort_feature_Y, ~] = sort(feature_db{item_num}(2,:));
    X1 = floor(sort_feature_X(1));
    Y1 = floor(sort_feature_Y(1));
    X2 = ceil(sort_feature_X(end));
    Y2 = ceil(sort_feature_Y(end));

    bounding_box_db{item_num} = [X1 X2 X2 X1; Y1 Y1 Y2 Y2];   
end
save('Item_database.mat', 'feature_db', 'descriptor_db', 'bounding_box_db', 'name_db', 'price_db');