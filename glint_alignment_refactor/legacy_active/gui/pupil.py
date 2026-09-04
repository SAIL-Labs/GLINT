import hcipy as hp
import numpy as np


## constants
PUPIL_DIAMETER = 7.95  # m
OBSTRUCTION_DIAMETER = 2.3397  # m
INNER_RATIO = OBSTRUCTION_DIAMETER / PUPIL_DIAMETER
SPIDER_WIDTH = 0.1735  # m
SPIDER_OFFSET = 0.639  # m, spider intersection offset
SPIDER_ANGLE = 51.75  # deg
ACTUATOR_SPIDER_WIDTH = 0.089  # m
ACTUATOR_SPIDER_OFFSET = (0.521, -1.045)
ACTUATOR_DIAMETER = 0.632  # m
ACTUATOR_OFFSET = ((1.765, 1.431), (-0.498, -2.331))  # (x, y), m
PUPIL_OFFSET = -41  # deg
PIXEL_SCALE = 5.77  # mas / pix

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def field_combine(field1, field2):
    return lambda grid: field1(grid) * field2(grid)




def generate_pupil_field(
    n: int = 256,
    outer: float = 1,
    inner: float = INNER_RATIO,
    scale: float = 1,
    angle: float = 0,
    oversample: int = 8,
    spiders: bool = True,
    actuators: bool = True,
):
    pupil_diameter = PUPIL_DIAMETER * outer
    # make grid over full diameter so undersized pupils look undersized
    max_diam = PUPIL_DIAMETER if outer <= 1 else pupil_diameter
    grid = hp.make_pupil_grid(n, diameter=max_diam)

    # This sets us up with M1+M2, just need to add spiders and DM masks
    # fix ratio
    inner_val = inner * PUPIL_DIAMETER
    inner_fixed = inner_val / pupil_diameter
    pupil_field = hp.make_obstructed_circular_aperture(pupil_diameter, inner_fixed)

    # add spiders to field generator
    if spiders:
        spider_width = SPIDER_WIDTH * scale
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))

        # spider in quadrant 1
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 2
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 3
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (-SPIDER_OFFSET, 0),  # start
                (-cost * pupil_diameter - SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )
        # spider in quadrant 4
        pupil_field = field_combine(
            pupil_field,
            hp.make_spider(
                (SPIDER_OFFSET, 0),  # start
                (cost * pupil_diameter + SPIDER_OFFSET, -sint * pupil_diameter),  # end
                spider_width=spider_width,
            ),
        )

    # add actuator masks to field generator
    if actuators:
        # circular masks
        actuator_diameter = ACTUATOR_DIAMETER * scale
        actuator_mask_1 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[0])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_1)

        actuator_mask_2 = hp.make_obstruction(
            hp.circular_aperture(diameter=actuator_diameter, center=ACTUATOR_OFFSET[1])
        )
        pupil_field = field_combine(pupil_field, actuator_mask_2)

        # spider
        sint = np.sin(np.deg2rad(SPIDER_ANGLE))
        cost = np.cos(np.deg2rad(SPIDER_ANGLE))
        actuator_spider_width = ACTUATOR_SPIDER_WIDTH * scale
        actuator_spider = hp.make_spider(
            ACTUATOR_SPIDER_OFFSET,
            (
                ACTUATOR_SPIDER_OFFSET[0] - cost * pupil_diameter,
                ACTUATOR_SPIDER_OFFSET[1] - sint * pupil_diameter,
            ),
            spider_width=actuator_spider_width,
        )
        pupil_field = field_combine(pupil_field, actuator_spider)

    rotated_pupil_field = hp.make_rotated_aperture(pupil_field, np.deg2rad(angle))

    return hp.evaluate_supersampled(rotated_pupil_field, grid, oversample)


def generate_pupil(*args, **kwargs):
    pupil = generate_pupil_field(*args, **kwargs)
    return pupil.shaped


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # pupil = generate_pupil(256, 1, 0.3, 1, 0, 8, True, True)
    pupil = generate_pupil()
    plt.imshow(pupil, origin="lower")
    plt.show()
generate_pupil.__doc__ = rf"""
Generate a SCExAO pupil parametrically.

Parameters
----------
n : int, optional
    Grid size in pixels. Default is 256
outer : float, optional
    Outer pupil diameter as a fraction of the true diameter. Default is 1.0
inner : float, optional
    Diameter of central obstruction as a fraction of the true diameter. Default is {INNER_RATIO:.03f}
scale : float, optional
    Scale factor for over-sizing spiders and actuator masks. Default is 1.0
angle : float, optional
    Pupil rotation angle, in degrees. Default is 0
oversample : int, optional
    Oversample factor for supersampling the pupil grid. Default is 8
spiders : bool, optional
    Add spiders to pupil. Default is True
actuators : bool, optional
    Add bad actuator masks and spider. Default is True

Notes
-----
The smallest element in the SCExAO pupil is the bad actuator spider, which is approximately {ACTUATOR_SPIDER_WIDTH*1e3:.1f} mm wide. This is about 0.7\% of the telescope diameter, which means you need to have a miinimum of ~142 pixels across the aperture to sample this element.

"""
# from skimage import transform
# from .centroid import cross_correlation_centroid, cutout_slice
# from scipy import optimize
# import matplotlib.pyplot as plt
# from matplotlib.colors import CenteredNorm

# def _model_func(params, psf):
#     # warp
#     tform = transform.EuclideanTransform(translation=params[3:])
#     shifted = transform.warp(psf, tform)
#     scaled = transform.rescale(shifted, params[0])
#     rotated = transform.rotate(scaled, params[1])
#     # clip again
#     inds = cutout_slice(rotated, window=psf.shape)
#     return params[2] * rotated[inds]

# def _resid_func(params, psf, image):
#     warped = _model_func(params, psf)
#     return warped - image

# def _loss_func(params, psf, image):
#     return np.nanmean(_resid_func(params, psf, image)**2)

# def fit_psf_scale_rotation(image, psf, plot=True):
#     ## Step 1: get cutout using cross-correlation registration
#     init_ctr = np.unravel_index(np.nanargmax(image), image.shape)
#     shift = cross_correlation_centroid(image, psf, init_ctr, return_shift=True)
#     tform = transform.EuclideanTransform(translation=shift)
#     image_reg = transform.warp(image, tform)
#     inds = cutout_slice(image_reg, window=psf.shape)
#     cutout = image_reg[inds]
#     target = cutout / np.nanmax(cutout)
#     psf_norm = psf / np.nanmax(psf)

#     P0 = [1, 0, 1, *shift]
#     result = optimize.minimize(_loss_func, P0, args=(psf_norm, target), method="Nelder-Mead", bounds=[(0.5, 1.5), (0, 360), (0, 2), (-10, 10), (-10, 10)])

#     if plot:
#         fig, axes = plt.subplots(1, 3)
#         axes[0].imshow(target, cmap="magma", origin="lower", norm="log")
#         axes[1].imshow(_model_func(result.x, psf_norm), cmap="magma", origin="lower", norm="log")
#         axes[2].imshow(_resid_func(result.x, psf_norm, target), origin="lower", cmap="bwr", norm=CenteredNorm())
#         axes[0].set_title("Target image")
#         axes[1].set_title("PSF image")
#         axes[2].set_title("Residual image")
#         fig.show()

#     return dict(scale=result.x[0], angle=result.x[1])