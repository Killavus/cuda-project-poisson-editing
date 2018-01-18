import numpy as np
import imageio
import scipy.ndimage as ndi
import sys

from optparse import OptionParser

def main():
  usage = "usage: %prog [options] bgimage pastedimage maskimage xpos ypos"

  parser = OptionParser(usage=usage)
  parser.add_option("-t", "--type",
                    dest="type",
                    help="Paste method to be used. Possible: naive, poisson, poisson_cg", default="naive")
  parser.add_option("-i", "--iterations",
                    dest="iterations",
                    help="How many iterations there should be for a given algorithm. For naive this option is ignored.", default=200)
  parser.add_option("-o", "--outfile",
                    dest="outfile",
                    help="Name of the file where output should be saved.",
                    default="out.png")
  options, args = parser.parse_args(sys.argv)
  args = args[1:]

  try:
    bgimagepath, pastedimagepath, maskimagepath, xpos, ypos = args
    run(bgimagepath, pastedimagepath, maskimagepath, xpos, ypos, options)
  except ValueError as e:
    print "Error: %s" % [e.message]
    parser.print_help()

def run(bgimagepath, pastedimagepath, maskimagepath, xpos, ypos, options):
  bg_image = (imageio.imread(bgimagepath)[:, :, :3] / 255.0).astype(np.float32)
  mask_image = imageio.imread(maskimagepath)[:, :, :3] > 0.5
  pasted_image = (imageio.imread(pastedimagepath)[:, :, :3] / 255.0).astype(np.float32)

  imageio.imwrite('debugfg.jpg', pasted_image)
 
  if options.type == 'naive':
    result = run_naive(
      bg_image,
      mask_image,
      pasted_image,
      int(xpos),
      int(ypos),
    )
  elif options.type == 'poisson':
    result = run_poisson(
      bg_image,
      mask_image,
      pasted_image,
      int(xpos),
      int(ypos),
      int(options.iterations)
    )
  elif options.type == 'poisson_cg':
    result = run_poisson_cg(
      bg_image,
      mask_image,
      pasted_image,
      int(xpos),
      int(ypos),
      int(options.iterations)
    )
  else:
    raise ValueError("Unknown paste method: %s" % options.type)

  imageio.imwrite(options.outfile, result)
  print "Saved result in %s." % options.outfile

def maskconv(img):
  return img / 255

def img256conv(img):
  return img / 255.0

def img1conv(img):
  return (img * 255.0).astype(np.uint8)

def run_naive(bg, mask, fg, x, y):
  h = min(fg.shape[0], bg.shape[0] - y)
  w = min(fg.shape[1], bg.shape[1] - x)
 
  result = np.copy(bg)
  composite = result[y:(y+h),x:(x+w)]
  composite[mask[:h,:w] != 0] = fg[mask[:h,:w] != 0]    
  
  return result

def im_dot(x, y):
  return (x * y).sum()

def laplacian_operator(img):
  kern = np.array([
    [ 0.0, -1.0,  0.0],
    [-1.0,  4.0, -1.0],
    [ 0.0, -1.0,  0.0]
  ])

  return np.dstack([
    ndi.convolve(img[:, :, 0], kern),
    ndi.convolve(img[:, :, 1], kern),
    ndi.convolve(img[:, :, 2], kern)
  ])

def run_poisson(bg, mask, fg, x, y, niter):
  print fg[0, :, :]
  h = min(fg.shape[0], bg.shape[0] - y)
  w = min(fg.shape[1], bg.shape[1] - x)

  big_fg = np.zeros_like(bg)
  big_mask = np.zeros(bg.shape).astype(mask.dtype)
  
  big_fg[y:(y+h),x:(x+w)] = fg
  big_mask[y:(y+h),x:(x+w)] = mask
  
  big_mask_3d = big_mask
  
  if big_mask_3d.shape[2] != 3:
      big_mask_3d = np.dstack([big_mask, big_mask, big_mask])
      
  big_mask_3d[big_mask_3d == 255] = 1
      
  cbg = bg.copy()
  cbg[big_mask_3d] = False
 
  b = laplacian_operator(big_fg)
  x = cbg + (big_fg * big_mask_3d)
  # imageio.imwrite('debugx.png', img1conv(x))
      
  for i in range(niter):
      r = (b - laplacian_operator(x)) * big_mask_3d
      a = im_dot(r, r) / im_dot(r, laplacian_operator(r))
      x = x + a * r
    
  return np.clip(x, 0, 1)

def run_poisson_cg(bg, mask, fg, x, y, niter):
  h = min(fg.shape[0], bg.shape[0] - y)
  w = min(fg.shape[1], bg.shape[1] - x)
  
  big_fg = np.zeros_like(bg)
  big_mask = np.zeros(bg.shape).astype(mask.dtype)
  
  big_fg[y:(y+h),x:(x+w)] = fg
  big_mask[y:(y+h),x:(x+w)] = mask
  
  big_mask_3d = big_mask
  
  if big_mask_3d.shape[2] != 3:
      big_mask_3d = np.dstack([big_mask, big_mask, big_mask])
              
  cbg = bg.copy()
  cbg[big_mask_3d] = False
          
  b = laplacian_operator(big_fg)
  x = cbg + (big_fg * big_mask_3d)
  r = (b - laplacian_operator(x)) * big_mask_3d
  d = r
      
  for i in range(niter):
      a = im_dot(r, r) / im_dot(d, laplacian_operator(d))
      x = x + a*d
      next_r = (r - a * laplacian_operator(d)) * big_mask_3d
      b = im_dot(next_r, next_r) / im_dot(r, r)
      r = next_r
      d = (r + b * d) * big_mask_3d

  return np.clip(x, 0, 1)

main()
