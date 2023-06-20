import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
def imshow( image ):
	plt.imshow(image)
	plt.axis('off');
	plt.show()

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('/Volumes/Samsung_T5/Graphics 2023/~gmu_graphics/mitsuba setup/scenes/cbox.xml', res=256, integrator='prb')
# image_ref = mi.render(scene, spp=128)
# render_bitmap = mi.Bitmap('render_small1.png')
render_bitmap = mi.Bitmap('render_small2.png').convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)

# render_bitmap.set_srgb_gamma(False)
image_ref = mi.TensorXf(render_bitmap)
# rfilter = mi.scalar_rgb.load_dict({'type': 'box'})
# image_ref = mi.TensorXf(render_bitmap.resample([128,128], rfilter))

print('xml',type(scene))
print('reference',type(image_ref))
# mi.util.write_bitmap("render.png", image_ref)

# Preview the reference image
imshow( mi.util.convert_to_bitmap(image_ref) )

params = mi.traverse(scene)

key = 'red.reflectance.value'

# Save the original value
param_ref = mi.Color3f(params[key])

# Set another color value and update the scene
# params[key] = mi.Color3f(0.01, 0.2, 0.9)
# params.update();

image_init = mi.render(scene, spp=128)
print('init',type(image_init))
imshow( mi.util.convert_to_bitmap(image_init) )

opt = mi.ad.Adam(lr=0.05)
opt[key] = params[key]
params.update(opt);

def mse(image):
	return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []
for it in range(iteration_count):
	image = mi.render(scene, params, spp=4) # render a rough sketch
	loss = mse(image) # calc loss by mean square error
	dr.backward(loss) # magic inverse rendering
	opt.step() # gradient step
	opt[key] = dr.clamp(opt[key], 0.0, 1.0) # clamps k-values between 0-1
	params.update(opt) # updates internal data structures
	err_ref = dr.sum(dr.sqr(param_ref - params[key]))
	print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
	errors.append(err_ref)
print('\nOptimization complete.')

image_final = mi.render(scene, spp=512)
imshow( mi.util.convert_to_bitmap(image_final) )

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()