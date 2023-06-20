import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt
def imshow( image ):
	plt.imshow(image)
	plt.axis('off');
	plt.show()

mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('/Volumes/Samsung_T5/Graphics 2023/~gmu_graphics/mitsuba setup/scenes/cbox.xml', res=128, integrator='prb')

# image_ref = mi.render(scene, spp=512)
image_ref = mi.Bitmap('render_edit.png')

# Preview the reference image
imshow( mi.util.convert_to_bitmap(image_ref) )

params = mi.traverse(scene)
print(params)

image_init = mi.render(scene, spp=128)
imshow( mi.util.convert_to_bitmap(image_init) )

opt = mi.ad.Adam(lr=0.05)
# key = 'red.reflectance.value'

# keys = ( 'red.reflectance.value', 'sphere3.to_world.scale.value', 'sphere3.to_world.translate.value' )
# for key in keys:
# 	opt[key] = params[key]

params.update(opt)

def mse(image):
	return dr.mean(dr.sqr(image - image_ref))

iteration_count = 3

errors = []
for it in range(iteration_count):
	image = mi.render(scene, params, spp=4)
	
	loss = mse(image)
	
	dr.backward(loss)
	
	opt.step()
	
	opt['red.reflectance.value'] = dr.clamp(opt[key], 0.0, 1.0)
	
	params.update(opt)
	
	# err_ref = dr.sum(dr.sqr(param_ref - params[key]))
	# print(f"Iteration {it:02d}: parameter error = {err_ref[0]:6f}", end='\r')
	print(f"Iteration {it:02d}: loss = {loss[0]:6f}", end='\r')
	errors.append(loss)
print('\nOptimization complete.')

image_final = mi.render(scene, spp=128)
imshow( mi.util.convert_to_bitmap(image_final) )

plt.plot(errors)
plt.xlabel('Iteration'); plt.ylabel('MSE(param)'); plt.title('Parameter error plot');
plt.show()
