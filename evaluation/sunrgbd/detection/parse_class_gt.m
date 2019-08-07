function [groundTruthBbs] = parse_class_gt(filename)
    % (YS)
    % The GT in the specified filename as stored in the format 
    % [img_id, cls_name, -1, -1, -10, box2d[0], box2d[1], box2d[2], 
    % box2d[3], h, w, l, tx, ty, tz, ry] in a .txt file.
    % [img_id, cls_name] will be parsed into fileData.textdata.
    % [-1, -1, -10, box2d[0], box2d[1], box2d[2], box2d[3], h, w, l,
    % tx, ty, tz, ry] will be parsed into fileData.data.
    %
    % This function parses the .txt file and stores the results in a
    % N_predictionsx1 struct array with fields 'centroid', 'basis',
    % 'coeffs', 'confidence'.
    
    format long;
    fileData = importdata(filename);
    N_gt = size(fileData.data, 1);
   
    allImgIds = cellfun(@str2num, fileData.textdata(:,1));
    groundTruthBbs(N_gt).classname   = '';
    groundTruthBbs(N_gt).imageNum    = 0;
    groundTruthBbs(N_gt).centroid    = [0,0,0];
    groundTruthBbs(N_gt).basis       = [0,0,0;0,0,0;0,0,0];
    % groundTruthBbs(N_gt).orientation = [0,0];
    groundTruthBbs(N_gt).coeffs      = [0,0,0];
    for i = 1:N_gt
        % Classname & Image number
        groundTruthBbs(i).classname = char(fileData.textdata(i,2));
        groundTruthBbs(i).imageNum  = allImgIds(i);
        
        % Center
        centroid = fileData.data(i,11:13);
        groundTruthBbs(i).centroid = [centroid(1) centroid(3) -centroid(2)];
        
        % Dimensions
        hwl_half = fileData.data(i,8:10) / 2.;
        lwh_half = fliplr(hwl_half);
        
%         wlh_half = [hwl_half(2), hwl_half(3), hwl_half(1)];
%         hlw_half = fliplr(wlh_half);

%         whl_half = [hwl_half(2), hwl_half(1), hwl_half(3)];
%         lhw_half = fliplr(whl_half);

        groundTruthBbs(i).coeffs = lwh_half;
        
        % Orientation
        heading_angle = fileData.data(i,14);
        groundTruthBbs(i).basis = rotz(radtodeg(-heading_angle))';
        % groundTruthBbs(i).orientation = [sin(heading_angle), cos(heading_angle)];
        
    end
    %groundTruthBbs = groundTruthBbs';

end