import pip


#Reference https://stackoverflow.com/questions/4527554/check-if-module-exists-if-not-install-it

def import_or_install(package):
    try:
        __import__(package)
    except ImportError:
        pip.main(['install', package])   
    
import_or_install("wget")
