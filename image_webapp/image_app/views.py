import os
import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render
from io import BytesIO
import base64
from django.http import JsonResponse
from django.http import HttpResponseServerError

global imd
def index(request):
    if request.method == 'POST':
        # Save the uploaded image
        uploaded_image = request.FILES['image']
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
        global imd 
        imd = uploaded_image.name
        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Get the selected noise type and noise level
        noise_type = request.POST.get('noise_type')
        noise_level = float(request.POST.get('noise_level'))

        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Apply the selected noise to the image
        noisy_image = apply_noise(image, noise_type, noise_level)

        # Save the noisy image
        noisy_image_path = os.path.join(settings.MEDIA_ROOT, 'noisy_image.png')
        cv2.imwrite(noisy_image_path, noisy_image)

        # Get the selected restoration technique
        restoration_technique = request.POST.get('restoration_technique')

        # Restore the noisy image using the selected technique
        restored_image = restore_image(noisy_image, restoration_technique)

        # Save the restored image
        restored_image_path = os.path.join(settings.MEDIA_ROOT, 'restored_image.png')
        cv2.imwrite(restored_image_path, restored_image)

        # Calculate the pixel histogram for comparison
        original_histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        restored_histogram = cv2.calcHist([restored_image], [0], None, [256], [0, 256])

        # Encode the images as base64 strings
        image_base64 = image_to_base64(image_path)
        noisy_image_base64 = image_to_base64(noisy_image_path)
        restored_image_base64 = image_to_base64(restored_image_path)

        # Prepare the data to be displayed in the template
        context = {
            'image_base64': image_base64,
            'noisy_image_base64': noisy_image_base64,
            'restored_image_base64': restored_image_base64,
            'original_histogram': original_histogram.flatten().tolist(),
            'restored_histogram': restored_histogram.flatten().tolist(),
        }

        # Render the index.html template with the result context
        return render(request, 'index.html', context)

    return render(request, 'index.html')

def apply_noise(image, noise_type, noise_level):
    # Add noise to the image based on the selected noise type and level
    if noise_type == 'gaussian':
        mean = 0
        var = noise_level ** 2
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, image.shape).astype(image.dtype)
        noisy_image = cv2.add(image, noise)
    elif noise_type == 'salt_and_pepper':
        p = noise_level
        noisy_image = image.copy()
        mask = np.random.choice([0, 1, 2], size=image.shape, p=[p / 2, p / 2, 1 - p])
        noisy_image[mask == 0] = 0  # Salt noise
        noisy_image[mask == 1] = 255  # Pepper noise
    elif noise_type == 'speckle':
        noise = noise_level * np.random.randn(*image.shape).astype(image.dtype)
        noisy_image = cv2.add(image, noise)
    else:
        noisy_image = image

    return noisy_image



def restore_image(image, restoration_technique):
    # Restore the image using the selected restoration technique
    if restoration_technique == 'gaussian_filter':
        restored_image = cv2.GaussianBlur(image, (5, 5), 0)
    elif restoration_technique == 'wiener_filter':
        psf = np.ones((5, 5)) / 25
        restored_image = cv2.filter2D(image, -1, psf)
    elif restoration_technique == 'median_filter':
        restored_image = cv2.medianBlur(image, 5)
    else:
        restored_image = image

    return restored_image


def image_to_base64(image_path):
    with open(image_path, 'rb') as file:
        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

    return image_base64

def delete_images(request):
    try:
        global imd
        # Delete the uploaded image, noisy image, and restored image
        image_path = os.path.join(settings.MEDIA_ROOT, imd)
        noisy_image_path = os.path.join(settings.MEDIA_ROOT, 'noisy_image.png')
        restored_image_path = os.path.join(settings.MEDIA_ROOT, 'restored_image.png')

        # Delete the images if they exist
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(noisy_image_path):
            os.remove(noisy_image_path)
        if os.path.exists(restored_image_path):
            os.remove(restored_image_path)

        return JsonResponse({'message': 'Images deleted successfully'})
    except Exception as e:
        return HttpResponseServerError(f'Failed to delete images: {str(e)}')