import sys
import subprocess 
import os

def install_dependencies():
    if '--user' in sys.argv:
        cmd = ['pip install -r requirements.txt --user']
    else:
        cmd = ['pip install -r requirements.txt']
    subprocess.call(cmd, shell=True)

def check_python_version():
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 5:
       return True
    return False




if __name__ == '__main__':
    import subprocess
    from setuptools import setup
    try:
        assert(check_python_version() )
    except AssertionError:
        sys.exit("Exiting: Please use python version > 3.6")
#    install_dependencies()

    exec(open('glopty/version.py').read())

    this_directory = os.path.abspath(os.path.dirname(__file__))

    with open(os.path.join(this_directory, 'README.rst')) as f:
        long_description = f.read()

    config = {'name':'glopty',
              'version':__version__,
              'description':'A global optimisation Module',
              'long_description': long_description,
              'long_description_content_type':"text/x-rst",
              'author':'Conn O\'Rourke',
     'author_email':'conn.orourke@gmail.com',
     'url':'https://github.com/connorourke/glopty',
     'python_requires':'>=3.6',
     'packages':['glopty'],
     'package_dir':{'glopty':'glopty'},
     'include_package_data':True,
     'license': 'MIT',
     'install_requires': ['numpy',
                          'pyDOE',
                          'scipy',
                          'pathos',
                          ]
}
    setup(**config)
