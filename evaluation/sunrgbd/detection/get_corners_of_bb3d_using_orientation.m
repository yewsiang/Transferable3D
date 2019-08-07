% Gets the 3D coordinates of the corners of a 3D bounding box.
%
% Args:
%   bb3d - 3D bounding box struct.
%
% Returns:
%   corners - 8x3 matrix of 3D coordinates.
%
% See:
%   create_bounding_box_3d.m
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function corners = get_corners_of_bb3d_using_orientation(bb3d)

  % Instead of using the basis vectors to calculate the 3D corners, we
  % simply use the orientation.
  heading_angle = -atan2(bb3d.orientation(2), bb3d.orientation(1));
  rot_matrix = rotz(radtodeg(heading_angle));
  coeffs = abs(bb3d.coeffs);
  corners = [-coeffs(1), coeffs(1), coeffs(1), -coeffs(1), -coeffs(1), coeffs(1), coeffs(1), -coeffs(1);
             coeffs(2), coeffs(2), -coeffs(2), -coeffs(2), coeffs(2), coeffs(2), -coeffs(2), -coeffs(2);
             coeffs(3), coeffs(3), coeffs(3), coeffs(3), -coeffs(3), -coeffs(3), -coeffs(3), -coeffs(3)];
  rot_corners = rot_matrix * corners;
  corners = rot_corners' + repmat(bb3d.centroid, [8 1]);
end
