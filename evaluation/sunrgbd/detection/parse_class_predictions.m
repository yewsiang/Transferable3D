function [allTestImgIds, allBb3dtight] = parse_class_predictions(filename)
    % (YS)
    % The predictions in the specified filename as stored in the format 
    % [img_id, cls_name, -1, -1, -10, box2d[0], box2d[1], box2d[2], 
    % box2d[3], h, w, l, tx, ty, tz, ry, score] in a .txt file.
    % [img_id, cls_name] will be parsed into fileData.textdata.
    % [-1, -1, -10, box2d[0], box2d[1], box2d[2], box2d[3], h, w, l,
    % tx, ty, tz, ry, score] will be parsed into fileData.data.
    %
    % This function parses the .txt file and stores the results in a
    % N_predictionsx1 struct array with fields 'centroid', 'basis',
    % 'coeffs', 'confidence'.
    
    format long;
    fileData = importdata(filename);
    N_predictions = size(fileData.data, 1);
   
    allTestImgIds = cellfun(@str2num, fileData.textdata(:,1));
    allBb3dtight(N_predictions).centroid    = [0,0,0];
    allBb3dtight(N_predictions).basis       = [0,0,0;0,0,0;0,0,0];
    allBb3dtight(N_predictions).coeffs      = [0,0,0];
    allBb3dtight(N_predictions).confidence  = 0;
    for i = 1:N_predictions
        % Dimensions
        hwl_half = fileData.data(i,8:10) / 2.;
        lwh_half = fliplr(hwl_half);
%          wlh_half = [hwl_half(2), hwl_half(3), hwl_half(1)];
%         hlw_half = fliplr(wlh_half);
%         whl_half = [hwl_half(2), hwl_half(1), hwl_half(3)];
%         lhw_half = fliplr(whl_half);
        allBb3dtight(i).coeffs = lwh_half;
        
        % Center
        centroid = fileData.data(i,11:13);
        allBb3dtight(i).centroid = [centroid(1) centroid(3) hwl_half(1)-centroid(2)];
        %allBb3dtight(i).centroid = [centroid(1) centroid(3) -centroid(2)];
        
        % Orientation
        heading_angle = fileData.data(i,14);
        allBb3dtight(i).basis = rotz(radtodeg(-heading_angle))';
        
        % Score
        allBb3dtight(i).confidence = fileData.data(i,15);
        
    end
    allBb3dtight = allBb3dtight';
end