
""" Visualization utilities.

Author: Tang Yew Siang
Date: July 2019
"""

import os
import vtk
import numpy as np
from PIL import Image, ImageDraw

##############################################################################
# General visualization utilities
##############################################################################

class Color:
  White      = (255,255,255)
  Black      = (0,0,0)
  Gray       = (179,179,179)
  DarkGray   = (60,60,60)
  LightGray  = (220,220,220)
  Green      = (0,120,0)
  DarkGreen  = (0,51,0)
  LightGreen = (0,255,0)
  Orange     = (255,156,28)
  Blue       = (0,102,255)
  Purple     = (255,0,255)
  Yellow     = (255,255,0)
  Red        = (255,0,0)

def convert_bbox_points_to_2D_rect_lines(x_coords, y_coords):
  """
  Given 8 2D points representing the vertices of a bbox,
  return a list of 4 4-tuples where each 4-tuple represents
  a line to be drawn to form a 2D bbox in the image.
  """
  min_x = np.min(x_coords)
  min_y = np.min(y_coords)
  max_x = np.max(x_coords)
  max_y = np.max(y_coords)
  return [(min_x, min_y, min_x, max_y), 
          (min_x, min_y, max_x, min_y), 
          (max_x, max_y, min_x, max_y), 
          (max_x, max_y, max_x, min_y)]

def get_image_with_bbox(img, box2Ds=[], width=0, color=Color.Red):
  """
  Draw 2D bboxes on the image.
  """
  # Display 2D Bbox if available
  if box2Ds != []:
    draw = ImageDraw.Draw(img)
    for box in box2Ds:
      assert(len(box) == 4)
      lines = convert_bbox_points_to_2D_rect_lines([box[0], box[2]], [box[1], box[3]])
      for line in lines:
        draw.line(line, fill=color, width=width)
  return img

def get_image_with_projected_bbox3d(img, proj_bbox3d_pts=[], width=0, color=Color.White):
  """
  Draw the outline of a 3D bbox on the image.
  Input:
    proj_bbox3d_pts: (8,2) array of projected vertices
  """
  v = proj_bbox3d_pts
  if proj_bbox3d_pts != []:
    draw = ImageDraw.Draw(img)
    for k in range(0,4):
      #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
      i,j = k,(k+1)%4
      draw.line([(v[i,0],v[i,1]), (v[j,0],v[j,1])], fill=color, width=width)

      i,j = k+4,(k+1)%4 + 4
      draw.line([(v[i,0],v[i,1]), (v[j,0],v[j,1])], fill=color, width=width)

      i,j = k,k+4
      draw.line([(v[i,0],v[i,1]), (v[j,0],v[j,1])], fill=color, width=width)
  return img

def get_image_with_points(img, pts=[], color=Color.White):
  """
  Draw 2D points on the image.
  Input:
    pts: list of (x,y)
  """
  if pts != []:
    draw = ImageDraw.Draw(img)
    # Convert inner list to tuple
    pts = [(x,y) for x,y in pts]
    draw.point(xy=pts, fill=color)
  return img

def display_image(filename, gt_box2Ds=[], gt_col=Color.LightGreen):
  img = Image.open(filename)
  img = get_image_with_bbox(img, box2Ds=gt_box2Ds, color=gt_col)
  img.show()

##############################################################################
# VTK visualization utilities
##############################################################################

class VtkPointCloud:
  """
  Visualizes a point cloud (colored by its labels) using vtk.
  """
  def __init__(self, points, gt_points=[], pred_points=[], use_rgb=False, color=Color.White):
    """
    points: (D,)
    gt_points (optional): (D,) binary values (0 or 1) where 1 means it belongs to object
    pred_points (optional): (D,) binary values (0 or 1) where 1 means it is predicted
    """
    if len(gt_points) > 0: assert(len(points) == len(gt_points))
    if len(pred_points) > 0: assert(len(points) == len(pred_points))

    self.vtk_poly_data = vtk.vtkPolyData()
    self.vtk_points = vtk.vtkPoints()
    self.vtk_cells = vtk.vtkCellArray()
    self.colors = vtk.vtkUnsignedCharArray()
    self.colors.SetNumberOfComponents(3)

    self.vtk_poly_data.SetPoints(self.vtk_points)
    self.vtk_poly_data.SetVerts(self.vtk_cells)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(self.vtk_poly_data)
    self.vtk_actor = vtk.vtkActor()
    self.vtk_actor.SetMapper(mapper)
    self.vtk_actor.GetProperty().SetPointSize(1.0)

    # Create point cloud 
    for i, point in enumerate(points):
      if use_rgb:
        rgb = np.array(point[3:6] * 255, dtype=np.int8)
        
        self.add_point(point, color=rgb)
        continue

      if len(gt_points) > 0 and len(pred_points) > 0:
        # If there is ground truth + prediction points
        if gt_points[i] == 1 and pred_points[i] == 1:
          # True positive
          self.add_point(point, color=Color.LightGreen)
        elif gt_points[i] == 0 and pred_points[i] == 0:
          # True negative
          self.add_point(point, color=color)
        elif gt_points[i] == 0 and pred_points[i] == 1:
          # False positive
          self.add_point(point, color=Color.Red)
        elif gt_points[i] == 1 and pred_points[i] == 0:
          # False negative
          self.add_point(point, color=Color.Yellow)
        else:
          raise Exception('Should not have such a situation')

      elif len(gt_points) > 0:
        # If there is only ground truth points
        if gt_points[i] == 1:
          self.add_point(point, color=Color.LightGreen)
        else:
          self.add_point(point, color=color)

      elif len(pred_points) > 0:
        # If there is only ground truth points
        if pred_points[i] == 1:
          self.add_point(point, color=Color.Orange)
        else:
          self.add_point(point, color=color)

      else:
        self.add_point(point, color=color)
 
  def add_point(self, point_with_label, color=Color.White):
    pointId = self.vtk_points.InsertNextPoint(point_with_label[0:3])
    self.vtk_cells.InsertNextCell(1)
    self.vtk_cells.InsertCellPoint(pointId)

    self.colors.InsertNextTuple3(color[0], color[1], color[2])
    self.vtk_poly_data.GetPointData().SetScalars(self.colors)
    self.vtk_poly_data.Modified()

    self.vtk_cells.Modified()
    self.vtk_points.Modified()

def vtk_lines(start_points, end_points, color=Color.White):
  assert(len(start_points) == len(end_points))
  colors = vtk.vtkUnsignedCharArray()
  colors.SetNumberOfComponents(3)
  colors.SetName("Colors")

  actors = []
  vtk_points = vtk.vtkPoints()
  for start, end in zip(start_points, end_points):
    vtk_points.InsertNextPoint(start)
    vtk_points.InsertNextPoint(end)

  N_lines = len(start_points)
  lines = vtk.vtkCellArray()
  for lineId in 2 * np.arange(N_lines):
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, lineId)
    line.GetPointIds().SetId(1, lineId + 1)
    lines.InsertNextCell(line)
    colors.InsertNextTuple(color)

  # Create a polydata to store everything in
  lines_poly_data = vtk.vtkPolyData()
  lines_poly_data.SetPoints(vtk_points)
  lines_poly_data.SetLines(lines)
  lines_poly_data.GetCellData().SetScalars(colors)

  # Visualize
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputData(lines_poly_data)

  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  return actor

def vtk_box_3D(points, line_width=1, color=Color.LightGreen):
  """
  3D bbox for display on the vtk visualization.
  """
  # Create a vtkPoints object and store the points in it
  vtk_points = vtk.vtkPoints()
  for pt in points:
    vtk_points.InsertNextPoint(pt)
   
  # Setup the colors array
  colors = vtk.vtkUnsignedCharArray()
  colors.SetNumberOfComponents(3)
  colors.SetName("Colors")
   
  lineIds = [(0, 1), (0, 3), (1, 2), (2, 3), # Lines in the bottom half
             (4, 5), (4, 7), (5, 6), (6, 7), # Lines in the top half
             (0, 4), (1, 5), (2, 6), (3, 7)] # Lines from bottom to top
  lines = vtk.vtkCellArray()
  for lineId in lineIds:
    line = vtk.vtkLine()
    line.GetPointIds().SetId(0, lineId[0])
    line.GetPointIds().SetId(1, lineId[1])
    lines.InsertNextCell(line)
    colors.InsertNextTuple(color)

  # Create a polydata to store everything in
  lines_poly_data = vtk.vtkPolyData()
  lines_poly_data.SetPoints(vtk_points)
  lines_poly_data.SetLines(lines)
  lines_poly_data.GetCellData().SetScalars(colors)
   
  # Visualize
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputData(lines_poly_data)
   
  actor = vtk.vtkActor()
  actor.SetMapper(mapper)
  actor.GetProperty().SetLineWidth(line_width)
  return actor

def vtk_image(img_filename, img_format='jpg', 
  box2Ds_list=[], box2Ds_cols=[], 
  pts2Ds_list=[], pts2Ds_cols=[],
  proj_box3Ds_list=[], proj_box3Ds_cols=[]):
  """
  Visualize the image at the given img_filename.
  box2Ds_list (m,n,4) is a list of (list of box2Ds that will be drawn on the image) where the ith index 
  is a (list of box2Ds) which will take the colour of the ith index of colors.
  pts2Ds_list (m,n,2) is a list of (list of 2D points that will be drawn on the image) where the ith index 
  is a (list of 2D pts) which will take the colour of the ith index of colors.
  proj_box3Ds_list (m,8,2) is a list of (list of projected 3D vertices that will be drawn on the image) 
  where the ith index is a (list of projected 3D pts) which will take the colour of the ith index of colors.
  """
  assert(len(box2Ds_list) == len(box2Ds_cols))
  assert(len(pts2Ds_list) == len(pts2Ds_cols))
  assert(len(proj_box3Ds_list) == len(proj_box3Ds_cols))

  img_pil = Image.open(img_filename)
  for box2Ds, box2Ds_col in zip(box2Ds_list, box2Ds_cols):
    img_pil = get_image_with_bbox(img_pil, box2Ds=box2Ds, color=box2Ds_col)

  for pts2Ds, pts2Ds_col in zip(pts2Ds_list, pts2Ds_cols):
    img_pil = get_image_with_points(img_pil, pts=pts2Ds, color=pts2Ds_col)

  for proj_box3Ds, proj_box3Ds_col in zip(proj_box3Ds_list, proj_box3Ds_cols):
    img_pil = get_image_with_projected_bbox3d(img_pil, proj_bbox3d_pts=proj_box3Ds, color=proj_box3Ds_col)

  import os
  img_dir = os.path.join('.', 'vis_tmp')
  if not os.path.exists(img_dir): os.mkdir(img_dir)
  img_path = os.path.join(img_dir, '1.png')
  img_pil.save(img_path)
  img = vtk.vtkPNGReader()
  img.SetFileName(img_path)
  img.Update()
  # os.remove(img_path)
  # os.rmdir(img_dir)

  actor = vtk.vtkImageActor()
  actor.GetMapper().SetInputConnection(img.GetOutputPort())
  return actor

def vtk_text(texts, arr_type='str', color=True, cols=8, sep=5, scale=(1,1,1), rot=(0,0,0), translate=(0,0,0)):
  """
  Visualizes text using vtk.
  If parse_and_color is set to True, it will parse the text and color "O" green and "X" red.
  """
  vtk_actors = []
  for n, text_str in enumerate(texts):
    x = (n % cols) * sep
    y = (n // cols) * sep
    
    if arr_type == 'float':
      if text_str < 0.333:
        col = (1,0,0)
      elif text_str < 0.666:
        col = (1,1,1)
      else:
        col = (0,1,0)
      if not color:
        col = (1,1,1)
      text_str = '%.1f%%' % (text_str * 100.)
    elif arr_type == 'boolean_list':
      text_arr = ['O' if boolean else 'X' for boolean in text_str]
    elif arr_type == 'text':
      col = (1,1,1)
    else:
      raise Exception('Unknown array type.')

    if arr_type == 'boolean_list':
      for k, boolean in enumerate(text_arr):
        text_source = vtk.vtkTextSource()
        col = (0,1,0) if boolean == 'O' else (1,0,0)
        text_source.SetForegroundColor(*col)
        text_source.SetText(boolean)
        text_mapper = vtk.vtkPolyDataMapper()
        text_mapper.SetInputConnection(text_source.GetOutputPort())

        additional_x_disp = k * 0.1
        text_actor = vtk.vtkActor()
        text_actor.SetMapper(text_mapper)
        trans = (translate[0] + x + additional_x_disp, translate[1] + y, translate[2])
        vtk_transform_actor(text_actor, scale=scale, rot=rot, translate=trans)
        vtk_actors.append(text_actor)
    else:
      text_source = vtk.vtkTextSource()
      text_source.SetForegroundColor(*col)
      text_source.SetText(text_str)
      text_mapper = vtk.vtkPolyDataMapper()
      text_mapper.SetInputConnection(text_source.GetOutputPort())
      
      text_actor = vtk.vtkActor()
      text_actor.SetMapper(text_mapper)
      trans = (translate[0] + x, translate[1] + y, translate[2])
      vtk_transform_actor(text_actor, scale=scale, rot=rot, translate=trans)
      vtk_actors.append(text_actor)
      
  return vtk_actors

def vtk_transform_actor(actor, scale=(1,1,1), rot=(0,0,0), translate=(0,0,0)):
  """
  Applies scaling, rotation or translation on a given vtk actor.
  """
  transform = vtk.vtkTransform()
  transform.PostMultiply()
  transform.Scale(*scale)
  transform.RotateX(rot[0])
  transform.RotateY(rot[1])
  transform.RotateZ(rot[2])
  transform.Translate(*translate)
  actor.SetUserTransform(transform)

class MyInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
  def __init__(self, ren_win_interactor, ren, ren_win, dict_key_to_actors_to_hide, ss_dir='vis_tmp'):
    """
    dict_key_to_actors_to_hide is a dictionary with a 'Keyboard key' to List of Vtk Actors mapping
    where if the particular keyboard key has been pressed, the corresponding Vtk Actors' visibility
    will be toggled.
    Can also be used to take screenshots into ss_dir by pressing 's'.
    """
    if not os.path.exists(ss_dir): os.mkdir(ss_dir)
    self.ss_dir = ss_dir
    self.ren = ren
    self.ren_win = ren_win
    self.ren_win_interactor = ren_win_interactor
    self.dict_key_to_actors_to_hide = dict_key_to_actors_to_hide
    self.AddObserver("KeyPressEvent",self.key_press_event)

  def key_press_event(self, obj, event):
    pressed_key = self.ren_win_interactor.GetKeySym()

    if pressed_key == 's':
      cam = self.ren.GetActiveCamera()
      # Choose the highest count of the 
      paths = [path for path in os.listdir(self.ss_dir) if path.split('_')[0] == 'screenshot']
      ss_count = max([int(path.split('_')[-1]) for path in paths]) + 1 if len(paths) > 0 else 1
      ss_name = 'screenshot_%03d' % ss_count
      print('\n--- Screenshot %s ---' % ss_name)
      print('pos = ' + str(cam.GetPosition()))
      print('fp = ' + str(cam.GetFocalPoint()))

      # Screenshot code
      screenshot_name = os.path.join(self.ss_dir, ss_name)
      w2if = vtk.vtkWindowToImageFilter()
      w2if.SetInput(self.ren_win)
      w2if.Update()
      writer = vtk.vtkPNGWriter()
      writer.SetFileName(screenshot_name)
      writer.SetInputData(w2if.GetOutput())
      writer.Write()

      return

    for key, actors in self.dict_key_to_actors_to_hide.items():
      if pressed_key == key:
        for actor in actors:
          visibility = actor.GetVisibility()
          actor.SetVisibility(1 - visibility)
        self.refresh()

  def refresh(self, resetCamera=False):
    if resetCamera:
      self.ren.ResetCamera()
    self.ren_win.Render()
    self.ren_win_interactor.Start()

def start_render(vtkActors, key_to_actors_to_hide={}, window_wh=(300,300), background_col=(0,0,0), background_col2=None, camera=None):
  """
  Start rendering a vtk actor.

  eyToActorsToHide is a dictionary with a 'Keyboard key' to List of Vtk Actors mapping
  where if the particular keyboard key has been pressed, the corresponding Vtk Actors' visibility
  will be toggled.

  Example:
    `start_render([vtkPC.vtkActor])`: Will render a point cloud without having any keypress events.
    `start_render([], { 'h': [vtkPC.vtkActor] })`: Will render a point cloud which will toggle its
      visibility if 'h' has been pressed.
  """
  # Renderer
  renderer = vtk.vtkRenderer()
  for vtkActor in vtkActors:
    renderer.AddActor(vtkActor)
  for key, actors in key_to_actors_to_hide.items():
    for actor in actors:
      renderer.AddActor(actor)

  background_col = (col / 255. for col in background_col)
  renderer.SetBackground(*background_col)
  if background_col2 is not None:
    renderer.GradientBackgroundOn()
    background_col2 = (col / 255. for col in background_col2)
    renderer.SetBackground2(*background_col2)

  if camera is not None:
    renderer.SetActiveCamera(camera)
  else:
    renderer.ResetCamera()
 
  # Setup render window, renderer, and interactor
  render_window = vtk.vtkRenderWindow()
  render_window.AddRenderer(renderer)
  render_window.SetSize(*window_wh)
  render_window_interactor = vtk.vtkRenderWindowInteractor()
  render_window_interactor.SetInteractorStyle(MyInteractorStyle(render_window_interactor, renderer, render_window, key_to_actors_to_hide))
  render_window_interactor.SetRenderWindow(render_window)
  render_window_interactor.Start()
  render_window.Render()
