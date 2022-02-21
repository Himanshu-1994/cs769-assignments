import os
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
    print("Package 'wget' does not exist. Please install")
    sys.exit()

url = "http://nlp.stanford.edu/data/glove.6B.zip"
filename = 'glove.6B.zip'
if not os.path.exists('glove.6B.zip'):
    print("Downloading Glove Embedding File")
    filename = wget.download(url)
    print("Done!")

print("Unzipping the contents of glove.6B.zip")
try:
    from zipfile import ZipFile
except ImportError:
    print("Package 'zipfile' does not exist. Please install")
    sys.exit()

ZipFile(filename).extract("glove.6B.300d.txt",".")

print("Done")
