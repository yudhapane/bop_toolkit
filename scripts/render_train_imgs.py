# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Renders RGB-D images of an object model."""

import os
import cv2

from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import view_sampler
import numpy as np
import matplotlib.pyplot as plt

views, views_level = view_sampler.sample_views(20)
count = 0
Views = np.zeros(shape=(len(views), 3))
x = Views[:,0]
y = Views[:,1]
z = Views[:,2]
ux = np.zeros(shape=(len(x), 1))
vx = np.zeros(shape=(len(x), 1))
wx = np.zeros(shape=(len(x), 1))
uy = np.zeros(shape=(len(x), 1))
vy = np.zeros(shape=(len(x), 1))
wy = np.zeros(shape=(len(x), 1))
uz = np.zeros(shape=(len(x),1))
vz = np.zeros(shape=(len(x),1))
wz = np.zeros(shape=(len(x),1))

# Color by azimuthal angle
cx = np.ones(shape=(len(x), 1))
cx = plt.cm.hsv(cx)

cy = np.ones(shape=(len(x), 1))
cy = plt.cm.brg(cy)

for view in views:
  Views[count, 0] = view["t"][0]
  Views[count, 1] = view["t"][1]
  Views[count, 2] = view["t"][2]
  ux[count] = view["R"][0,0]
  vx[count] = view["R"][0,1]
  wx[count] = view["R"][0,2]
  uy[count] = view["R"][1,0]
  vy[count] = view["R"][1,1]
  wy[count] = view["R"][1,2]
  uz[count] = -view["R"][2,0]
  vz[count] = -view["R"][2,1]
  wz[count] = -view["R"][2,2]
  count = count + 1



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.clear()
ax.scatter3D(Views[:, 0], Views[:, 1], Views[:, 2], cmap='Greens')
ax.quiver(x.transpose(), y.transpose(), z.transpose(), ux.transpose(), vx.transpose(), wx.transpose(), length=0.1, normalize=True, colors=cx)
ax.quiver(x.transpose(), y.transpose(), z.transpose(), uy.transpose(), vy.transpose(), wy.transpose(), length=0.1, normalize=True, colors=cy)
ax.quiver(x.transpose(), y.transpose(), z.transpose(), uz.transpose(), vz.transpose(), wz.transpose(), length=0.1, normalize=True)
plt.show()


# x, y, z = np.meshgrid(Views[:,0],Views[:,1],Views[:,2])
# u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
# v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
# w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
#      np.sin(np.pi * z))


# PARAMETERS.
################################################################################
# See dataset_params.py for options.
dataset = 'tless'

# Radii of view spheres from which to render the objects.
if dataset == 'lm':
  radii = [400]  # There are only 3 occurrences under 400 mm.
elif dataset == 'tless':
  radii = [650]
elif dataset == 'tudl':
  radii = [850]
elif dataset == 'tyol':
  radii = [500]
elif dataset == 'ruapc':
  radii = [590]
elif dataset == 'icmi':
  radii = [500]
elif dataset == 'icbin':
  radii = [450]
else:
  raise ValueError('Unknown dataset.')

# Type of object models and camera.
model_type = None
cam_type = None
if dataset == 'tless':
  model_type = 'reconst'
  cam_type = 'primesense'

# Objects to render ([] = all objects from the specified dataset).
obj_ids = []

# Minimum required number of views on the whole view sphere. The final number of
# views depends on the sampling method.
min_n_views = 1000

# Rendering parameters.
ambient_weight = 0.5  # Weight of ambient light [0, 1]
shading = 'phong'  # 'flat', 'phong'

# Type of the renderer. Options: 'vispy', 'cpp', 'python'.
renderer_type = 'vispy'

# Super-sampling anti-aliasing (SSAA) - the RGB image is rendered at ssaa_fact
# times higher resolution and then down-sampled to the required resolution.
# Ref: https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
ssaa_fact = 4

# Folder containing the BOP datasets.
datasets_path = config.datasets_path

# Folder for the rendered images.
out_tpath = os.path.join(config.output_path, 'render_{dataset}')

# Output path templates.
out_rgb_tpath =\
  os.path.join('{out_path}', '{obj_id:06d}', 'rgb', '{im_id:06d}.png')
out_depth_tpath =\
  os.path.join('{out_path}', '{obj_id:06d}', 'depth', '{im_id:06d}.png')
out_scene_camera_tpath =\
  os.path.join('{out_path}', '{obj_id:06d}', 'scene_camera.json')
out_scene_gt_tpath =\
  os.path.join('{out_path}', '{obj_id:06d}', 'scene_gt.json')
out_views_vis_tpath =\
  os.path.join('{out_path}', '{obj_id:06d}', 'views_radius={radius}.ply')
################################################################################


out_path = out_tpath.format(dataset=dataset)
misc.ensure_dir(out_path)

# Load dataset parameters.
dp_split_test = dataset_params.get_split_params(datasets_path, dataset, 'test')
dp_model = dataset_params.get_model_params(datasets_path, dataset, model_type)
dp_camera = dataset_params.get_camera_params(datasets_path, dataset, cam_type)

if not obj_ids:
  obj_ids = dp_model['obj_ids']

# Image size and K for the RGB image (potentially with SSAA).
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp_camera['im_size']]
K_rgb = dp_camera['K'] * ssaa_fact

# Intrinsic parameters for RGB rendering.
fx_rgb, fy_rgb, cx_rgb, cy_rgb =\
  K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]

# Intrinsic parameters for depth rendering.
K = dp_camera['K']
fx_d, fy_d, cx_d, cy_d = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

# Create the RGB renderer.
width_rgb, height_rgb = im_size_rgb[0], im_size_rgb[1]
ren_rgb = renderer.create_renderer(
  width_rgb, height_rgb, renderer_type, mode='rgb', shading=shading)
ren_rgb.set_light_ambient_weight(ambient_weight)

# Add object models to the RGB renderer.
for obj_id in obj_ids:
  ren_rgb.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))

# Create the depth renderer.
width_depth, height_depth,  = dp_camera['im_size'][0], dp_camera['im_size'][1]
ren_depth = renderer.create_renderer(
  width_depth, height_depth, renderer_type, mode='depth')

# Add object models to the depth renderer.
for obj_id in obj_ids:
  ren_depth.add_object(obj_id, dp_model['model_tpath'].format(obj_id=obj_id))

# Render training images for all object models.
for obj_id in obj_ids:

  # Prepare output folders.
  misc.ensure_dir(os.path.dirname(out_rgb_tpath.format(
    out_path=out_path, obj_id=obj_id, im_id=0)))
  misc.ensure_dir(os.path.dirname(out_depth_tpath.format(
    out_path=out_path, obj_id=obj_id, im_id=0)))

  # Load model.
  model_path = dp_model['model_tpath'].format(obj_id=obj_id)
  model = inout.load_ply(model_path)

  # Load model texture.
  if 'texture_file' in model:
    model_texture_path =\
      os.path.join(os.path.dirname(model_path), model['texture_file'])
    model_texture = inout.load_im(model_texture_path)
  else:
    model_texture = None

  scene_camera = {}
  scene_gt = {}
  im_id = 0
  for radius in radii:
    # Sample viewpoints.
    view_sampler_mode = 'hinterstoisser'  # 'hinterstoisser' or 'fibonacci'.
    views, views_level = view_sampler.sample_views(
      min_n_views, radius, dp_split_test['azimuth_range'],
      dp_split_test['elev_range'], view_sampler_mode)

    misc.log('Sampled views: ' + str(len(views)))
    # out_views_vis_path = out_views_vis_tpath.format(
    #   out_path=out_path, obj_id=obj_id, radius=radius)
    # view_sampler.save_vis(out_views_vis_path, views, views_level)

    # Render the object model from all views.
    for view_id, view in enumerate(views):
      if view_id % 10 == 0:
        misc.log('Rendering - obj: {}, radius: {}, view: {}/{}'.format(
          obj_id, radius, view_id, len(views)))

      # Rendering.
      rgb = ren_rgb.render_object(
        obj_id, view['R'], view['t'], fx_rgb, fy_rgb, cx_rgb, cy_rgb)['rgb']
      depth = ren_depth.render_object(
        obj_id, view['R'], view['t'], fx_d, fy_d, cx_d, cy_d)['depth']

      # Convert depth so it is in the same units as other images in the dataset.
      depth /= float(dp_camera['depth_scale'])

      # The OpenCV function was used for rendering of the training images
      # provided for the SIXD Challenge 2017.
      rgb = cv2.resize(rgb, dp_camera['im_size'], interpolation=cv2.INTER_AREA)
      # rgb = scipy.misc.imresize(rgb, par['cam']['im_size'][::-1], 'bicubic')

      # Save the rendered images.
      out_rgb_path = out_rgb_tpath.format(
        out_path=out_path, obj_id=obj_id, im_id=im_id)
      inout.save_im(out_rgb_path, rgb)
      out_depth_path = out_depth_tpath.format(
        out_path=out_path, obj_id=obj_id, im_id=im_id)
      inout.save_depth(out_depth_path, depth)

      # Get 2D bounding box of the object model at the ground truth pose.
      # ys, xs = np.nonzero(depth > 0)
      # obj_bb = misc.calc_2d_bbox(xs, ys, dp_camera['im_size'])

      scene_camera[im_id] = {
        'cam_K': dp_camera['K'],
        'depth_scale': dp_camera['depth_scale'],
        'view_level': int(views_level[view_id])
      }

      scene_gt[im_id] = [{
        'cam_R_m2c': view['R'],
        'cam_t_m2c': view['t'],
        'obj_id': int(obj_id)
      }]

      im_id += 1

  # Save metadata.
  inout.save_scene_camera(out_scene_camera_tpath.format(
    out_path=out_path, obj_id=obj_id), scene_camera)
  inout.save_scene_gt(out_scene_gt_tpath.format(
    out_path=out_path, obj_id=obj_id), scene_gt)
