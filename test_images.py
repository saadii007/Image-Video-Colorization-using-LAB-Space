import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
    return images

# Step 1: Load black and white images and corresponding true color images
bw_images = load_images_from_folder("images/")
color_images = load_images_from_folder("colored_images/")

# Step 2: Load the Caffe model and cluster points
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("model/colorization_deploy_v2.prototxt", "model/colorization_release_v2.caffemodel")
pts = np.load("model/pts_in_hull.npy")

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Initialize lists to store accuracy metrics
mse_scores = []
psnr_scores = []
ssim_scores = []

# Step 3: Compute accuracy metrics for each black and white image
for bw_image, true_color_image in zip(bw_images, color_images):
    # Scale the pixel intensities to the range [0, 1] and convert to Lab color space
    scaled = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the Lab image to 224x224, split channels, extract the 'L' channel, and perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Pass the L channel through the network which will predict the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted 'ab' volume to the same dimensions as the input image
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

    # Grab the 'L' channel from the original input image and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert the output image from the Lab color space to RGB and clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Compute accuracy metrics
    mse = mean_squared_error(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB), cv2.cvtColor(true_color_image, cv2.COLOR_BGR2RGB))
    
    # Convert colorized and true color images to uint8 data type and scale to [0, 255] range
    colorized_uint8 = np.clip((colorized * 255).astype(np.uint8), 0, 255)
    true_color_image_uint8 = np.clip((true_color_image * 255).astype(np.uint8), 0, 255)
    
    # Compute PSNR using the uint8 images
    psnr = peak_signal_noise_ratio(cv2.cvtColor(colorized_uint8, cv2.COLOR_BGR2RGB), cv2.cvtColor(true_color_image_uint8, cv2.COLOR_BGR2RGB))
    
    ssim = structural_similarity(cv2.cvtColor(colorized_uint8, cv2.COLOR_BGR2GRAY), cv2.cvtColor(true_color_image_uint8, cv2.COLOR_BGR2GRAY), data_range=255)
    
    mse_scores.append(mse)
    psnr_scores.append(psnr)
    ssim_scores.append(ssim)

# Step 4: Calculate average accuracy metrics
avg_mse = sum(mse_scores) / len(mse_scores)
avg_psnr = sum(psnr_scores) / len(psnr_scores)
avg_ssim = sum(ssim_scores) / len(ssim_scores)

# Step 5: Calculate accuracy percentages
total_images = len(bw_images)
accuracy_percentage_mse = (sum(mse <= 5000 for mse in mse_scores) / total_images) * 100
accuracy_percentage_psnr = (sum(psnr >= 30 for psnr in psnr_scores) / total_images) * 100
accuracy_percentage_ssim = (sum(ssim >= 0.5 for ssim in ssim_scores) / total_images) * 100

# Step 6: Display results
print("Average MSE:", avg_mse)
print("Average PSNR:", avg_psnr)
print("Average SSIM:", avg_ssim)
print("Accuracy Percentage (MSE): {:.2f}%".format(accuracy_percentage_mse))
print("Accuracy Percentage (PSNR): {:.2f}%".format(accuracy_percentage_psnr))
print("Accuracy Percentage (SSIM): {:.2f}%".format(accuracy_percentage_ssim))

# Step 7: Plot accuracy metrics over the dataset
plt.figure(figsize=(15, 5))

# Plot MSE
plt.subplot(1, 3, 1)
plt.plot(mse_scores, color='blue')
plt.xlabel('Image Index')
plt.ylabel('MSE')
plt.title('MSE over Dataset')

# Plot PSNR
plt.subplot(1, 3, 2)
plt.plot(psnr_scores, color='green')
plt.xlabel('Image Index')
plt.ylabel('PSNR')
plt.title('PSNR over Dataset')

# Plot SSIM
plt.subplot(1, 3, 3)
plt.plot(ssim_scores, color='red')
plt.xlabel('Image Index')
plt.ylabel('SSIM')
plt.title('SSIM over Dataset')

plt.tight_layout()
plt.show()