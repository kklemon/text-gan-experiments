from setuptools import setup

setup(name='vq_vae_text',
      version='0.0.1',
      packages=['vq_vae_text'],
      install_requires=['torch>=0.4.0',
                        'torchvision>=0.2.0',
                        'numpy']
)
