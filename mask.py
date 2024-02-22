from PIL import Image, ImageDraw
import os

# Define paths to your dataset
dataset_dir = 'C:\\Users\Defargobeze\Desktop\Ambo\Dataset/images'
alphabet_dirs = os.listdir(dataset_dir)

# Define output directory for masks
output_masks_dir = 'C:\\Users\Defargobeze\Desktop\Ambo\Dataset/masks'


# Define mask color
mask_color = (255, 255, 255)  # White color for masks

# Loop through each alphabet directory
for alphabet_dir in alphabet_dirs:
    alphabet_images_dir = os.path.join(dataset_dir, alphabet_dir)
    image_files = os.listdir(alphabet_images_dir)
    alphabet_masks_dir = os.path.join(output_masks_dir, alphabet_dir)
    if not os.path.exists(alphabet_masks_dir):
      os.makedirs(alphabet_masks_dir)
    # Create a mask for each image in the alphabet directory
    for image_file in image_files:
        image_path = os.path.join(alphabet_images_dir, image_file)

        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Create a new blank mask for the image
        mask = Image.new('RGB', image.size, color=(0, 0, 0))
        draw = ImageDraw.Draw(mask)

        # Manually annotate the mask according to the alphabet's shape
        # For example, you can draw the shape of the alphabet with white color on the mask
        # This step may require manual intervention or a segmentation tool
        # Here's a simple example assuming you're drawing a rectangle as a mask
        draw.rectangle([(10, 10), (100, 100)], fill=mask_color)

        # Save the mask with the same filename as the image in the output masks directory
        mask_filename = os.path.splitext(image_file)[0] + '.jpg'
        mask_path = os.path.join(alphabet_masks_dir, mask_filename)
        mask.save(mask_path)

        print(f"Mask created for {image_file}")

print("Mask creation completed.")
