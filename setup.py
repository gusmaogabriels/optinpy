from distutils.core import setup


requires = ['numpy']

packages = [
		'optinpy',
		'optinpy.graph',
		'optinpy.finitediff',
		'optinpy.linesearch',
    		'optinpy.mcfp',
    		'optinpy.mst',
    		'optinpy.nonlinear',
		'optinpy.nonlinear.constrained',
		'optinpy.nonlinear.unconstrained',
   		'optinpy.simplex',
    		'optinpy.sp',
]

package_dir = {'optinpy' : 'optinpy'}
package_data = { 'optinpy' : []}


setup(
    name='optinpy',
    version='1.0.0',
    packages=packages,
    license='The  GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007 License',
    author = 'Gabriel S. Gusmao',
    author_email = 'gusmaogabriels@gmail.com',
    url = 'https://github.com/gusmaogabriels/optinpy',
    download_url = 'https://github.com/gusmaogabriels/xl2py/tarball/v1.0.0',
    keywords = ['python', 'optimization', 'linear', 'nonlinear', 'graph', 'node', 'arc'],
    package_data = package_data,
    package_dir = package_dir,

)
