"""
    Written by Jonny Hyman, 2020
        www.jonnyhyman.com
        www.github.com/jonnyhyman

This code generates and renders 3D volumetric data of the
Mandelbrot set.

Note that volumetric rendering is very resource intensive so rendering out to
f rames might be the only option for most normal computers (not interactive)

Code structure is roughly:
    - imports
    - functions
    - parameters
    - runtime stuff

"""

from vispy.color import get_colormaps, BaseColormap
from vispy import app, scene, io
import vispy.visuals.volume
import vispy

from numba import jit, njit, prange
from time import time
import vispy.io as io
import numpy as np
import math

from pathlib import Path

import os

# ---- FUNCTIONS

def ffmpeg():
    """ Render the frames we have created into a ProRes mov"""

    global rec_prefix, project_name, dropbox_folder

    convert_cmd = (f"""ffmpeg -f image2 -framerate 24"""
                       f""" -i {rec_prefix}/{project_name}/{project_name}_%d.png"""
                       f""" -c:v prores_ks -profile:v 3 {dropbox_folder}/{project_name}.mov""")

    print('CONVERTING >>>', convert_cmd)

    os.system(convert_cmd)

    dir = (f"./{rec_prefix}/{project_name}")

    print('REMOVING >>>', dir)

    filelist = [f for f in os.listdir(dir) if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join(dir, f))

    print(f"{project_name} is rendered")

    quit()


@jit(nopython=True)
def filter3d_core(image, filt, result):
    """ Core of 3D smoothing algorithm"""
    M, N, O = image.shape
    Mf, Nf, Of = filt.shape
    Mf2 = Mf // 2
    Nf2 = Nf // 2
    Of2 = Of // 2

    for i in range(Mf2, M - Mf2):
        for j in range(Nf2, N - Nf2):
            for k in range(Of2, O - Of2):
                num = 0
                for ii in range(Mf):
                    for jj in range(Nf):
                        for kk in range(Of):
                            num += (filt[Mf-1-ii, Nf-1-jj, Of-1-kk] * image[i-Mf2+ii, j-Nf2+jj, k-Of2+kk])
                result[i, j, k] = num


@jit(nopython=True)
def filter3d(image, filt):
    """ 3D smoothing algorithm to reduce Moire patterns """
    result = np.zeros_like(image)
    filter3d_core(image, filt, result)
    return result

@njit(cache=True)
def linear_interp(x, in_min, in_max, out_min, out_max):
    """ linear interpolation function
        maps `x` between `in_min` -> `in_max`
        into `out_min` -> `out_max`
    """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

@njit(cache=True)
def pix2point(shape, ix, iy, ovs=1):
    """ find the nearest pixel to point `ix`, `iy`
        in the plane with shape `shape`
        ovs is the oversampling factor for spatial antialiasing
    """

    cr = ((ix/ovs)*3.0/shape[0] - 2.0) # [0, width] > [-2, 1]
    ci = ((iy/ovs)*2.0/shape[1] - 1.0) # [0, height] > [-1, 1]

    return cr, ci

@njit(cache=True, parallel=True)
def make_mandelbrot(width, height, max_iterations):
    """
        Create a 2D Mandelbrot set to use as a mask when we create the
        3D mandelbrot dataset

        Can be faster if parallel=False for small data sizes

    """

    result = np.zeros((width, height))

    # for each pixel at (ix, iy)
    for iy in prange(height):
        for ix in prange(width):

            # start iteration at x0 in [-2, 1] and y0 in [-1, 1]
            x0, y0 = pix2point(result.shape, ix,iy)

            x = 0.0
            y = 0.0
            # perform Mandelbrot set iterations
            for iteration in range(max_iterations):

                # complex number multiplication
                # z = z**2 + c
                # z = x+yi
                # c = x0 + y0i
                x_new = x*x - y*y + x0
                y = 2*x*y + y0
                x = x_new

                # if escaped
                if x*x + y*y > 4.0:
                    # color using pretty linear gradient
                    color = 1.0
                    break
                else:
                    # failed, set color to black
                    color = 0.0

            result[ix, iy] = color

    return result

@njit(cache=True)
def antialias(x,y,z,n,m,o):
    """
        Core of spatial antialiasing
    """

    i_nmo0 = int(round(x + n))
    i_nmo1 = int(round(y + m))
    i_nmo2 = int(round(z + o))

    ic0 = 1 - abs(x - (i_nmo0 + 0.5))
    ic1 = 1 - abs(y - (i_nmo1 + 0.5))
    ic2 = 1 - abs(z - (i_nmo2 + 0.5))

    return (ic0 * ic1 * ic2), i_nmo0, i_nmo1, i_nmo2

@njit(cache=True, parallel=True)
def make_triplebrot(mask, width, height, depth, max_iterations,
                                    cutoff=500, oversample=1):
    """
    Create the 3D dataset of Mandelbrot set / logistic map, dubbed `triplebrot`

    mask : precomputed 2D mandelbrot set to quickly exclude points not in
            the mandelbrot set

    width, height, depth : x, y, z data dimensions

    oversample : to decrease aliasing, we further subdivide x/y/z dimensions
                by oversample to get more datapoints into the set



    Iterates a typical mandelbrot set, but keeps z real iterates
        x = c real
        y = c imag
        z = z real
    """

    result = np.zeros((width, height, depth), dtype=np.float64)
    nmo_range = np.array([-1,0,+1], dtype=np.int32)

    # for each pixel at (ix, iy)
    for iy in prange(height*oversample):
        for ix in prange(width*oversample):

            if mask[ix//oversample, iy//oversample] != 0:
                # not in the mandelbrot set
                continue

            cr, ci = pix2point(result.shape, ix,iy, ovs = oversample)

            zr = 0.0
            zi = 0.0

            # perform Mandelbrot set iterations
            for iteration in range(max_iterations):

                # complex number multiplication
                # z = z**2 + c
                # z = x+yi
                # c = x0 + y0i

                zr_new = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = zr_new

                # if escaped
                if zr*zr + zi*zi > 4.0:
                    break

                if iteration <= cutoff:
                    continue

                x = result.shape[0]/3.0 * (cr+2.0) #  [-2, 1] > [0, width]
                y = result.shape[1]/2.0 * (ci+1.0) # [-1, 1] > [0, height]
                z = result.shape[2]/4.0 * (zr+2.0) #  [-2, 2] > [0, depth]

                # blend into every neighboring voxel : spatial antialising

                for n in nmo_range:
                    for m in nmo_range:
                        for o in nmo_range:

                            d, i_nmo0, i_nmo1, i_nmo2 = antialias(x,y,z,n,m,o)

                            if d > 1 or d <= 0:
                                continue

                            if (1 <= i_nmo0 < result.shape[0] and
                                1 <= i_nmo1 < result.shape[1] and
                                1 <= i_nmo2 < result.shape[2]):

                                d = max(0, d)
                                d = min(1, d)

                                # 1e-6 is an arbitrary "small" number
                                # can actually be anything as long as
                                # the eventual value doesn't exceed the max
                                # possible 64bit floating point value

                                result[i_nmo0, i_nmo1, i_nmo2] += 1e-6 * d

    # here we reiterate exactly what we just did, but only along the x-z plane
    # this enhances the logistic map which would otherwise be too faint

    for iy in prange(result.shape[1]//2 - 1, result.shape[1]//2 +1):
        for ix in prange(width*oversample):

            if mask[ix//oversample, iy//oversample] != 0:
                continue

            cr, ci = pix2point(result.shape, ix,iy, ovs = oversample)

            zr = 0.0
            zi = 0.0

            # perform Mandelbrot set iterations
            for iteration in range(max_iterations):

                # complex number multiplication
                # z = z**2 + c
                # z = x+yi
                # c = x0 + y0i

                zr_new = zr*zr - zi*zi + cr
                zi = 2*zr*zi + ci
                zr = zr_new

                # if escaped
                if zr*zr + zi*zi > 4.0:
                    break

                if iteration <= cutoff:
                    continue

                x = result.shape[0]/3.0 * (cr+2.0) #  [-2, 1] > [0, width]
                y = result.shape[1]/2.0 * (ci+1.0) # [-1, 1] > [0, height]
                z = result.shape[2]/4.0 * (zr+2.0) #  [-2, 2] > [0, depth]

                d, i_nmo0, i_nmo1, i_nmo2 = antialias(x,y,z,0,0,0)

                result[i_nmo0, i_nmo1, i_nmo2] += 1e-6

    return result

# ---- PARAMETERS

f = 0 # start frame

F = 24*30 // 2 # frames per camera move

D = 100 # dimensions of all sides of the volume cube
# D = 1000 in final rendered animation

Omax = 1 # oversampling
# Omax = 16 in final rendered animation

#Omax seems to scale as: time to render in seconds = (3*o)**2.7 @ D=1000
# on a 64 core machine at 2.7GHz per core

I = 1500 # maximum iterations in the mandelbrot calculation

# rec_size = (2538, 1080) # frame resolution, that we used in the video
rec_size = (1280, 720) # a better resolution for most screens

# where to put the frames
rec_prefix = './frames'
project_name = f'fractal_mandelbrot_{D}'
frame_dir = Path(f'{rec_prefix}/{project_name}')

# Record frames?
rec = False

# Playback keyframes?
play = True

# Generate data?
generate = False


# ---- RUNTIME

data_prefix = './data'
datafile = f'{data_prefix}/fractal_mandelbrot_data{D}_iter{I}_ovs{Omax}.npz'

if not Path(rec_prefix).exists():
    Path(rec_prefix).mkdir()

if not Path(data_prefix).exists():
    Path(data_prefix).mkdir()

if not frame_dir.exists() and rec:
    frame_dir.mkdir()

if not generate:
    # Read volume
    vol = np.load(datafile)['data']

else:

    start = time()
    mset = make_mandelbrot(D, D, I)
    print("Made mandelbrot importance mask", time()-start)

    start = time()

    #for O in np.arange(2,Omax):
    O = int(round(Omax))
    # scales as: seconds = (3*o)**2.7

    print("STARTED... D =",D,"O =",O)
    if D == 1000:
        print('ETA...', (100*(3*O/10)**2.7)/(60*60),'hrs' )

    print(f"{project_name} at {D},{I},{O} started generating")

    vol = make_triplebrot(mset, D, D, D, I, oversample=O)
    print('DONE', time()-start)

    GB = vol.size * vol.itemsize * 1e-9
    print("SAVING GIGABYTES = ", GB)
    np.savez(datafile, data=vol)
    print("DUSTED!")

    print(f"{project_name} at {D},{I},{O} is done and dusted"
                f" {GB}GB after {time()-start}")

# Draw x,y,z axes into the data
#vol[:,0,0] = np.linspace(0,vol.max(),vol.shape[0])
#vol[0,:,0] = np.linspace(0,vol.max(),vol.shape[1])
#vol[0,0,:] = np.linspace(0,vol.max(),vol.shape[2])

# Data filtering / smoothing seems to reduce too much detail
# so it is deprecated here
#print("filtering")
#start = time()
#vol = filter3d(vol, np.ones( (2,)*3 ))
#print(f"done filtering! {(time()-start)/D**3} sec / voxel")


# Increase the visibility of the x-z plane (where the logistic map lies)
vol[:,vol.shape[1]//2,:] = np.sqrt(vol[:,vol.shape[1]//2,:])

# Slice the data to reveal only the logistic map
slice = 0

if slice:
    vol = vol[:,vol.shape[1]//2 -1 : vol.shape[1]//2 + 2,:]

# Prepare canvas
canvas = scene.SceneCanvas(keys='interactive', size=rec_size, show=not rec)

# Set up a viewbox to display the image with interactive pan/zoom
view = canvas.central_widget.add_view()

stepsize = .1  # step size inside the fragment shader : 0.1 is highest quality

# Modify the vispy `transclucent` volume OpenGL shader

NEW_TRANSLUCENT_SNIPPETS = dict(
    before_loop="""vec4 integrated_color = vec4(0., 0., 0., 0.); """,
    in_loop    =("""color = $cmap(val);
float a1 = integrated_color.a;

// MODIFIED color.r -> val because we have no alpha info
float a2 = val * (1 - a1);

float alpha = max(a1 + a2, 0.001);

// Doesn't work.. GLSL optimizer bug?
//integrated_color = (integrated_color * a1 / alpha) +
//                   (color * a2 / alpha);
// This should be identical but does work correctly:
integrated_color *= a1 / alpha;
integrated_color += color * a2 / alpha;

integrated_color.a = alpha;

if( alpha > 0.99 ){
    // stop integrating if the fragment becomes opaque
    iter = nsteps;
}"""),
    after_loop="""gl_FragColor = integrated_color;""",
)

class NewVolume(scene.visuals.Volume):
    def __init__(self, *args, **kwargs):
        self._rendering_methods['translucent'] = NEW_TRANSLUCENT_SNIPPETS
        super().__init__(*args, **kwargs)

# Create the volume visuals, only one is visible
volume1 = NewVolume(vol, parent = view.scene,
                                cmap = 'nipy_spectral_r', # colormap
                                method = 'translucent', # shader method
                                relative_step_size = stepsize)



# rescale the volume to look like a `normal` mandelbrot set
volume1.transform = scene.MatrixTransform()
volume1.transform.scale([1,2/3,1])
volume1.transform.rotate(90, [0,1,0])

# translate the entire volume to be in the middle
if not slice:
    volume1.transform.translate([0, 1/6 * vol.shape[1], vol.shape[0]])
else:
    volume1.transform.translate([0, (2/3-.4)*vol.shape[1], vol.shape[0]])


cam = scene.cameras.TurntableCamera(parent=view.scene, fov=2.0,
                                     name='Turntable')

# Start distance (empircally determined)
Ds = 5000 * vol.shape[0]/100

view.camera = cam
view.camera.elevation = 90
view.camera.azimuth = 0
view.camera.distance = Ds

@njit
def smooth(f, start, end):
    """ Camera smoothing function; essentially an ease in / ease out
        nonlinear interpolation
    """
    f = 0.5*(np.cos(np.pi*(f-1)) + 1) # assumes x between 0-1

    return linear_interp(f, 0,1, start, end)

def camera_move(camera, f, F):

    # which keyframe we are on = phase
    phase = f//F

    # where we are between current keyframe and next (0-1) = fader
    fader = (f % F)/F

    # camera keyframes; assumes all lists equal length
    keyframes = {

        'elevation' : [ 0,
                        0,
                        90,
                        0,
                        -45,
                    ],

        'distance'  : [ Ds/2,
                        Ds/2,
                        Ds/1.5,
                        Ds/1.5,
                        Ds/1.5,
                    ],

        'azimuth'   : [ 0,
                        180,
                        180,
                        90,
                        0,
                    ],

    }

    if phase >= len(keyframes['azimuth'])-1:
        # we've reached the end of the keyframes (assumes all equal length)
        if rec:
            ffmpeg()

    #print('phase:',phase,'fader:',fader)

    for k in keyframes:
        try:

            val = smooth(fader, keyframes[k][phase],
                                 keyframes[k][phase + 1])
            setattr(view.camera, k, val)

        except IndexError:
            print('length mismatch for', k)

    return len(keyframes['azimuth']) * F

def update(event):
    global view, f, F, vol, volume1, canvas, rec, rec_prefix, project_name, \
            play

    start = time()

    if not play:
        if rec:
            play = True

    if play:
        maxF = camera_move(view.camera, f, F)

    f += 1

    if rec:

        image = canvas.render()
        io.write_png(f'{rec_prefix}/{project_name}/{project_name}_{f}.png', image)

        ETA = (time() - start) * (maxF-f) # (time / frame) * frames remaining
        ETA = (ETA / 60) / 60 # seconds to hours
        ETA = np.modf(ETA)
        ETA = int(ETA[1]), int(round(ETA[0]*60))
        ETA = str(ETA[0]) + ":" + str(ETA[1]).zfill(2)

        print('saved frame', f, 'eta:', ETA)

# Implement axis connection with cam
@canvas.events.mouse_move.connect
def on_mouse_move(event):
    pass


# Implement key presses
@canvas.events.key_press.connect
def on_key_press(event):
    pass


if __name__ == '__main__':

    print(f"{project_name} started rendering")
    a = app.Timer(connect=update, start=True, app=canvas.app)
    app.run()
