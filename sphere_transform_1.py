import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
from mitsuba.scalar_rgb import Transform4f as T

def imshow( image ):
	plt.imshow(image)
	plt.axis('off');
	plt.show()

mi.set_variant('llvm_ad_rgb')

# ref image
scene = mi.load_file('scenes2/cbox.xml', res=256, integrator='prb_reparam')
render_bitmap = mi.Bitmap('render1transform.png').convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, srgb_gamma=True)
image_ref = mi.TensorXf(render_bitmap)
print("show ref image")
imshow(image_ref)

og = mi.render(scene, spp=128)
mi.util.write_bitmap("aaa.png", og)

# opt image
image_init = mi.render(scene, spp=128)
print("show start image")
imshow( mi.util.convert_to_bitmap(image_init) )


params = mi.traverse(scene)
keys = ['3sphere.to_world']
print(type(params[keys[0]]))
print('initial T', params[keys[0]])


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
	
	print('T', T)

	params['3sphere.to_world'] = T
	params.update()

def mse(image):
	return dr.mean(dr.sqr(image - image_ref))

iteration_count = 50

errors = []
for it in range(iteration_count):
	apply_transformation( params, opt )
	
	image = mi.render(scene, params, spp=4) # render a rough sketch
	
	# save intermidate image
	
	inter_filename = "render_it_" + str(it) + ".png"
	mi.util.write_bitmap(inter_filename, image)
	
	loss = mse(image) # calc loss by mean square error
	dr.backward(loss) # magic inverse rendering
	opt.step() # gradient step
	
	params.update(opt)

	print(f"Iteration {it:02d}: loss = {loss[0]:6f}")	
	errors.append(loss)

print('\nOptimization complete.')

image_final = mi.render(scene, spp=512)
imshow( mi.util.convert_to_bitmap(image_final) )

mi.util.write_bitmap('final_image.png', image_final)



plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()