import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T

def imshow( image ):
	plt.imshow(image)
	plt.axis('off');
	plt.show()

mi.set_variant('llvm_ad_rgb')

integrator_dict = mi.load_dict({
	'type': 'direct_reparam',
	'reparam_rays': 8
})
integrator_dict
# mi.register_integrator("reparam", lambda props: integrator_dict(props))

scene = mi.load_file('scenes2/cbox.xml', res=256, integrator='prb_reparam')
render_bitmap = mi.Bitmap('render1transform.png').convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)

image_ref = mi.TensorXf(render_bitmap)
mi.util.write_bitmap("render1.png", image_ref)
params = mi.traverse(scene)

keys = ['3sphere.to_world']
print(type(params[keys[0]]))

# Save the original values
param_refs = [] 
for key in keys:
	param_refs.append(params[key])

image_init = mi.render(scene, spp=128)
# print('init',type(image_init))
mi.util.write_bitmap("render1.png", image_init)
imshow( mi.util.convert_to_bitmap(image_init) )

# Adam optimizer
opt = mi.ad.Adam(lr=0.05)
opt['sphere1.translation'] = mi.Vector3f(0,0,0)
opt['sphere1.axis'] = mi.Vector3f(0.0,0.0,1.0)
opt['sphere1.angle'] = mi.Float(0.0)
opt['sphere1.scale'] = mi.Float(1.0)

def apply_transformation(params, opt):
    T = ( mi.Transform4f
	     .translate(opt['sphere1.translation'])
		 .rotate(opt['sphere1.axis'], opt['sphere1.angle'])
		 .scale(opt['sphere1.scale'])
		 )

    params['3sphere.to_world'] = T
    params.update()

def mse(image):
	return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []
for it in range(iteration_count):
	apply_transformation( params, opt )
	image = mi.render(scene, params, spp=4) # render a rough sketch
	mi.util.write_bitmap("render1out.png", image) # view the output after each iteration
	loss = mse(image) # calc loss by mean square error
	dr.backward(loss) # magic inverse rendering
	opt.step() # gradient step

	print(f"Iteration {it:02d}: ", loss)
	errors.append(loss)
	if len( errors ) >= 2 and abs( errors[-1][0] - errors[-2][0] ) < 1e-7:
		print( f"Converged after {len(errors)} iterations due to tiny change in loss value." )
		break
print('\nOptimization complete.')

image_final = mi.render(scene, spp=512)
mi.util.write_bitmap("render1out.png", image_final)
imshow( mi.util.convert_to_bitmap(image_final) )

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()