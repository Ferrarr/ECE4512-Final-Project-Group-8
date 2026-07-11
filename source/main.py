from Extractor import extract
from Classifier import classify
from Enhance import enhance
import cv2 as cv
import os
from datetime import datetime

# def main():
#     image = "0002a5b67e5f0909_jpg.rf.c8f81ef986e3e99af6f349c200080453.jpg"
#     plates, confidences = extract(image)

#     for plate in plates:
#         degradations = classify(plate)

#         enhanced_plate = enhance(plate, degradations)

#         # save image to output/ folder.
#         os.makedirs("output", exist_ok=True)
#         filename = "output/enhanced_image_" + str(datetime.now().strftime("%H-%M-%S-%f")) + ".jpg"
#         cv.imwrite(filename, enhanced_plate)


# For the time being im using this code first 
#  As we still doesnt have the classify 
# And apparently in my case "Motion Blur"
# does not really need to be cropped 
# using the Extractor 

def main(): 
    imagePath = input("Please input the Image Source Path \n>>> ")
    image = cv.imread(imagePath, cv.IMREAD_GRAYSCALE)
    if image is None: 
        print(f'Failed to load ({imagePath})')
        return None 
    degradation = input('Please input Degradation Type (Select a Number)\n(1) Motion Blur \n(2) Noise \n>>> ')
    intDeg = int(degradation)
    degradationType = None 
    match intDeg: 
        case 1: 
            degradationType = 'motionBlur'
        case 2: 
            degradationType = 'noise'
        case _: 
            print('Unknown Degradation Type')
            return None 
        
    recoveredImage = enhance(image, degradationType)
    

# Easy terminal UI 
# Put the directory for the image
# (Remember, relative to the source)
#  e.g "../assets/motionBlurred/1.png"
# and then use user input (Integer)
# to select algorithm 

main()
