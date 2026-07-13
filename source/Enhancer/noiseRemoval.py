# MASIH JELEK BANGET, I NEED TO IMPLEMENT MUCH BETTER ALGO 

import matplotlib.pyplot as plt 
import numpy as np 
# Salt and pepper Noise
def medianFilter(image, kernelSize): 
    height, width =  image.shape[:2]
    padSize = kernelSize // 2
    ph, pw = height + (2 * padSize), width + (2 * padSize)
    paddedImg = np.zeros((ph, pw), dtype=np.uint8)
    paddedImg[padSize: padSize+ height, padSize:padSize+width] = image 

    resultImage = np.zeros((height, width), dtype = np.uint8)
    for i in range(padSize, padSize + height): 
        for j in range(padSize, padSize + width): 
            window = paddedImg[i - padSize: i+ padSize+1, j-padSize:j+padSize+1]
            sortedPixel = np.sort(window.flatten())
            medianVal = sortedPixel[len(sortedPixel) // 2]
            resultImage[i-padSize, j-padSize] = medianVal

    return resultImage

# Gaussian Noise 
def meanFilter(image, kernelSize): 
    height, width = image.shape[:2]
    resultImage = np.zeros((height, width), dtype= np.uint8)
    padSize = kernelSize // 2
    ph, pw = height + (2 * padSize), width + (2 * padSize)
    paddedImage = np.zeros((ph, pw), dtype = np.uint8)
    paddedImage[padSize:padSize + height, padSize:padSize+width] = image 

    for i in range(padSize, padSize + height): 
        for j in range(padSize, padSize + width): 
            window = paddedImage[i - padSize : i + padSize + 1, 
                                 j - padSize : j + padSize + 1]
            meanVal = np.round(np.mean(window)).astype(np.uint8)
            resultImage[i-padSize, j-padSize] = meanVal

    return resultImage
            
def removeNoise(image): 
    while True: 
        noiseType = input('Please select the type of noise inside of the picture \n(1) Salt and Pepper \n(2) Gaussian Noise \n>>> ')
        try: 
            noiseType = int(noiseType)
            if (noiseType < 1 or noiseType > 2): 
                print('input invalid! Please choose between 1 or 2')
                continue 
            break 
        except: 
            print('Please Enter a valid input!')
            continue 

    lastRestoredImg = None
    lastKernelSize = None 
    restorationProcess = None 

    while True: 
        filterRes = input('Please enter kernelSize (must be odd) \nWrite down "OK" if you are okay with current configuration\n(Write "exit" to terminate)\n>>> ')
        if filterRes == 'OK': 
            if lastRestoredImg is None: 
                print('You must configure ur setting atleast once')
                continue 
            return lastRestoredImg, lastKernelSize, restorationProcess
        if filterRes == 'exit' : 
            print("Session terminated")
            break
        try: 
            filterRes = int(filterRes)
        except ValueError: 
            print('Please enter a valid input (Integer) or "OK"!')
            continue 
        if filterRes < 3: 
            print('The Kernel Size should be at least 3!')
            continue 
            
        if filterRes % 2 == 0: 
            print('The kernel size is EVEN, Auto Adjusting implemented.') 
            filterRes = filterRes - 1
            
        
        match noiseType: 
            case 1: 
                restoredImage = medianFilter(image, filterRes)
                restorationProcess = 'Median Filter'
            case 2: 
                restoredImage = meanFilter(image, filterRes)
                restorationProcess = 'Mean Filter'
            case _: 
                print('Input invalid!')
                return None
        lastRestoredImg = restoredImage
        lastKernelSize = filterRes
        plt.figure(figsize=(6, 6)) 
        plt.suptitle("Restored Image")
        plt.title(f'{restorationProcess} | Kernel Size: {filterRes}')
        plt.axis('off')
        plt.imshow(restoredImage, cmap='gray') 
        plt.show()
    
    
    