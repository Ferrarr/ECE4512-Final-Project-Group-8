# I put all of the long code inside of the folder 
# "Enhancer" and im just going to import it here
# im thinking that this way will make 
#  the code to be much more neat 
from Enhancer.estimateMotionBlur import estimateMotionBlur
from Enhancer.noise import removeNoise

# function to brighten image in case of low light
def brighten(image):
    return 1


# function to deblur image in case of blurred image
def deblur(image):
    return 1


# this function name sucks, we can change it later
def motionBlurRestore(image):
    estimateMotionBlur(image)
    return 1


def derain(image):
    return 1


def super_resolution(image):
    return 1


def depixelate(image):
    return 1


def unglare(image):
    return 1


def denoise(image):
    removeNoise(image) 
    return 1

def enhance(image, degradations):
    match degradations: 
        case 'motionBlur': 
            motionBlurRestore(image)
        case 'noise': 
            denoise(image)
        case _: 
            print(f'Unknown Info: {image}, {degradations}')
            return None

    return 1
