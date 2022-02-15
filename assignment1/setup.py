import pip
import sys

#Reference https://stackoverflow.com/questions/4527554/check-if-module-exists-if-not-install-it

def import_or_install(package):
    try:
        __import__(package)
        print("successfully imported - ",package)
        return 1
    except ImportError:
        pip.main(['install', package])   
    return 0

package = "wget"
status = import_or_install(package)

if not status:
    try:
        __import__(package)
    except ImportError:
        print("Package: ",package, "not installed")
        sys.exit()
