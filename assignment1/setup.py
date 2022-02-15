import pip
import sys

#Reference https://stackoverflow.com/questions/4527554/check-if-module-exists-if-not-install-it

def import_or_install(package):
    try:
        __import__(package)
        print("Successfully imported package: ",package)
    except ImportError:
        pip.main(['install', package])   

package = 'wget'
status = import_or_install(package)

try:
    import wget
except ImportError:
    print("wget not installed")
    sys.exit()

url = "http://nlp.stanford.edu/data/glove.6B.zip"
filename = wget.download(url)
print(filename)
